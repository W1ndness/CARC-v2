import os.path
from abc import ABCMeta
from abc import abstractmethod

import sklearn.base
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import d2l
import d2l.torch


class Classifier(metaclass=ABCMeta):
    def __init__(self, class_, model, labels, model_name):
        if class_ not in ['sklearn', 'torch', 'tf']:
            raise ValueError(class_, f'{class_} is not supported.')
        self.model = model
        self.__class = class_
        self.labels = labels
        self.model_name = model_name

    @abstractmethod
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y, sample_weight=None):
        pass


class SklearnClassifier(Classifier):
    def __init__(self, model: sklearn.base.ClassifierMixin, labels, model_name):
        super().__init__('sklearn', model, labels, model_name)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.model.score(X, y, sample_weight=sample_weight)


class TorchClassifier(Classifier):  # @device just CPU
    def __init__(self, model: torch.nn.Module, labels, model_name,
                 num_epochs: int,
                 criterion: torch.nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int):
        super().__init__('torch', model, labels, model_name)
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.devices = d2l.torch.try_all_gpus()
        self.model_param_path = f'{self.model_name}_params.pth'
        self.model_path = f'{self.model_name}_model.pth'

    @staticmethod
    def __train_batch(model, X, y, loss, trainer, devices):
        if isinstance(X, list):
            X = [x.to(devices[0]) for x in X]
        else:
            X = X.to(devices[0])
        y = y.to(devices[0])
        model.train()
        trainer.zero_grad()
        pred = model(X)
        l = loss(pred, y)
        l.sum().backward()
        trainer.step()
        train_loss_sum = l.sum()
        train_acc_sum = d2l.accuracy(pred, y)
        return train_loss_sum, train_acc_sum

    @staticmethod
    def __train(model, train_iter, test_iter,
                loss, trainer,
                num_epochs,
                devices):
        timer, num_batches = d2l.torch.Timer(), len(train_iter)
        animator = d2l.torch.Animator(xlabel='epoch',
                                      xlim=[1, num_epochs], ylim=[0, 1],
                                      legend=['train loss', 'train acc', 'test acc'])
        model = torch.nn.DataParallel(model, device_ids=devices).to(devices[0])
        for epoch in range(num_epochs):
            # 4个维度：储存训练损失，训练准确度，实例数，特点数
            metric = d2l.torch.Accumulator(4)
            for i, (features, labels) in enumerate(train_iter):
                timer.start()
                l, acc = TorchClassifier.__train_batch(
                    model, features, labels, loss, trainer, devices)
                metric.add(l, acc, labels.shape[0], labels.numel())
                timer.stop()
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (metric[0] / metric[2], metric[1] / metric[3],
                                  None))
            test_acc = d2l.torch.evaluate_accuracy_gpu(model, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
              f'{str(devices)}')

    @staticmethod
    def init_weights(m):
        if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
            torch.nn.init.xavier_uniform_(m.weight)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.model.apply(TorchClassifier.init_weights)
        train_iter = DataLoader(TensorDataset(X_train, y_train))
        test_iter = DataLoader(TensorDataset(X_test, y_test))
        TorchClassifier.__train(self.model,
                                train_iter, test_iter,
                                self.criterion,
                                self.optimizer,
                                self.num_epochs,
                                self.devices)
        torch.save(self.model, self.model_path)
        torch.save(self.model.state_dict(), self.model_param_path)

    @staticmethod
    def __infer(model, X, devices):
        with torch.no_grad():
            X = X.to(devices[0])
            label = torch.squeeze(model(X))
        return label.cpu().numpy()[0]

    def predict(self, X_eval):
        if not os.path.exists(self.model_param_path):
            raise Exception("Model has not been trained.")
        self.model.load_state_dict(torch.load(self.model_param_path))
        self.model.to(self.devices[0])
        preds = []
        for X in X_eval:
            pred = TorchClassifier.__infer(self.model, X)
            preds.append(pred)
        return preds
