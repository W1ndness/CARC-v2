import os
import sys
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from abc import ABCMeta
from abc import abstractmethod

import sklearn.base
import torch.nn
import joblib
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from d2l import torch as d2l


class Classifier(metaclass=ABCMeta):
    def __init__(self, class_, model, labels, model_name):
        if class_ not in ['sklearn', 'torch', 'tf']:
            raise ValueError(class_, f'{class_} is not supported.')
        self.model = model
        self.__class = class_
        self.labels = labels
        self.model_name = model_name

    @abstractmethod
    def fit(self, X_train, y_train, X_test, y_test):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y, sample_weight=None):
        pass


class SklearnClassifier(Classifier):
    def __init__(self, model: sklearn.base.ClassifierMixin, labels, model_name):
        """
        Init method for a sklearn-based classifier
        :param model: a sklearn-based classifier
        :param labels: labels for classification
        :param model_name: the name of the classifier, hopes to be unique
        """
        super().__init__('sklearn', model, labels, model_name)
        self.model_path = f'{self.model_name}_model.pth'

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        print("Score on train samples:", self.score(X_train, y_train))
        print("Score on test samples:", self.score(X_test, y_test))
        joblib.dump(self.model, self.model_path)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.model.score(X, y, sample_weight=sample_weight)


class TorchClassifier(Classifier):  # @save

    def __init__(self, model: torch.nn.Module, labels, model_name,
                 dataset_class,
                 num_epochs: int,
                 criterion: torch.nn.modules.loss._Loss,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int):
        """
        Init method for a torch-based classifier
        :param model: a torch-based classifier, usually as MLP
        :param labels: labels for classification
        :param model_name: the name of the classifier, hopes to be unique
        :param dataset_class: the class wrapping the data(usually as np.ndarray)
        This argument firstly designed for BERT embedding. Normally, the value is torch.TensorDataset,
        but clftools.datasets.bert.BertDataset for BERT. You can set another class for your data.
        :param num_epochs: the number of epochs for training
        :param criterion: usually called loss function
        :param optimizer: the optimizer for net
        :param batch_size: training batch size
        """
        super().__init__('torch', model, labels, model_name)
        self.dataset_class = dataset_class
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.devices = d2l.try_all_gpus()
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
        timer, num_batches = d2l.Timer(), len(train_iter)
        animator = d2l.Animator(xlabel='epoch',
                                xlim=[1, num_epochs], ylim=[0, 1],
                                legend=['train loss', 'train acc', 'test acc'])
        model = torch.nn.DataParallel(model, device_ids=devices).to(devices[0])
        for epoch in range(num_epochs):
            # 4个维度：储存训练损失，训练准确度，实例数，特点数
            metric = d2l.Accumulator(4)
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
            test_acc = d2l.evaluate_accuracy_gpu(model, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        animator.fig.show()
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
              f'{str(devices)}')

    @staticmethod
    def init_weights(m):
        if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
            torch.nn.init.xavier_uniform_(m.weight)

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.apply(TorchClassifier.init_weights)
        print("====== Data Shape ======")
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape", y_test.shape)
        print("========================")
        X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
        X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
        train_iter = DataLoader(self.dataset_class(X_train, y_train), batch_size=self.batch_size)
        test_iter = DataLoader(self.dataset_class(X_test, y_test), batch_size=self.batch_size)
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
            digits = torch.squeeze(model(X))
        label = digits.cpu().numpy().argmax()
        return label

    def predict(self, X_eval):
        if not os.path.exists(self.model_param_path):
            raise Exception("Model has not been trained.")
        self.model.load_state_dict(torch.load(self.model_param_path))
        self.model.to(self.devices[0])
        preds = []
        for X in X_eval:
            pred = TorchClassifier.__infer(self.model, X, self.devices)
            preds.append(pred)
        return preds

    def score(self, X, y, sample_weight=None):
        pass
