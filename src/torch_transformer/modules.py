import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def clone_layers(module, num_layers):
    """
    Replicate `num_layers` of type module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

def subsequent_mask(size):
    """
    Mask out subsequent layers of encoder/decoder for self attention
    """
    attn_shape = (1, size, size) # ???
    # np triu returns a triangular matrix with elements below kth diagonal zeroed
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 # == 0 will interchange 0's and 1's

def batch_size_fn(new, count, sofar):
    """
    Augmenting batch and calculate total number of tokens + padding
    """
    # is global really necessary?
    # global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt)+2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def random_data_gen(vocab, batch_size, n_batches):
    """
    Generate random data for a src-tgt copy task.
    """
    for i in range(n_batches):
        data = torch.from_numpy(np.random.randint(1, vocab, size=(batch_size, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class Batch:
    """
    Object for holding a batch of data with mask during training
    """
    def __init__(self, src, tgt=None, pad=0):
        super(Batch, self).__init__()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # difference between tgt and tgt_y?
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # Why are n_tokens? tokens = vocab item?
            self.n_tokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        mask to hide padding and future words
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class DecayingOptimizer:
    """
    A decaying Adam optimizer whose learning rate decays overtime
    lrate = factor * (d_model)^-0.5*min(steps^-0.5, steps*(warmup_steps)^-1.5)
    warmup_steps = 4000 if total 300k
    beta_1 = 0.9
    beta_2 = 0.98
    epsilon = 1e-9
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0

    def step(self):
        """
        Update parameters and rate every step
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Calculate new learning rate
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size**(-0.5)*min(step**(-0.5), step * self.warmup**(-1.5)))

    @staticmethod
    def get_optimizer(model):
        return DecayingOptimizer(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) 


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing using the KL div loss. Instead of using a one-hot
    target distribution, create a distribution that has confidence of the
    correct word and the rest of the smoothing mass distributed throughout the
    vocabulary.
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        print("size", size)
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    """
    Simple loss computation and training
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        # ??? why norm?
        print("norm in loss", norm)
        print("x", x)
        print("y", y)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)/norm)
        print("loss back before")
        loss.backward()
        print("loss back after")
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        print("loss item", loss.data.item())
        print("norm", norm)
        return loss.data.item() * norm.data.item()


