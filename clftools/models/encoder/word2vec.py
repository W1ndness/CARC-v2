from torch import nn
from d2l import torch as d2l
import torch


class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


class Word2VecNet(nn.Module):
    def __init__(self, vocab, embed_size):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)


class Word2VecEncoder:
    def __init__(self, vocab, embed_size):
        self.vocab = self.vocab
        self.embed_size = self.embed_size
        self.net = Word2VecNet(vocab, embed_size)

    @staticmethod
    def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
        v = embed_v(center)
        u = embed_u(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

    def train(self, data_iter, lr, num_epochs, device=d2l.try_gpu()):
        def init_weights(m):
            if type(m) == nn.Embedding:
                nn.init.xavier_uniform_(m.weight)

        self.net.train()
        self.net.apply(init_weights)
        net = self.net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                xlim=[1, num_epochs])
        # 规范化的损失之和，规范化的损失数
        metric = d2l.Accumulator(2)
        for epoch in range(num_epochs):
            timer, num_batches = d2l.Timer(), len(data_iter)
            for i, batch in enumerate(data_iter):
                optimizer.zero_grad()
                center, context_negative, mask, label = [
                    data.to(device) for data in batch]

                pred = Word2VecEncoder.skip_gram(center, context_negative,
                                                 self.net.in_embed, self.net.out_embed)
                loss = SigmoidBCELoss()
                l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
                l.sum().backward()
                optimizer.step()
                metric.add(l.sum(), l.numel())
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}, '
              f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

    def embedding(self, token):
        W = self.net.in_embed.weight.data
        return W[self.vocab[token]]



