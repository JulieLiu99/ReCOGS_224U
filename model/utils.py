import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor 
from jaxtyping import Shaped
from typing import Callable

def chamferToken(loss_fn: Callable , 
                 a: Shaped[Tensor, "bs nt nl vs"], 
                 b: Shaped[Tensor, "bs nt nl"], 
                 mask_a: Shaped[Tensor, "bs nt nl"], 
                 mask_b: Shaped[Tensor, "bs nt nl"],
                 reduce: bool=True,
                 split=False):
    """
    chamfer dist for tokens
    loss_fn: loss function that computes distance between tokens
    a: [bs, nt, nl, vocab_size] number of tokens, max length
    b: [bs, nt, nl]
    mask_a, mask_b: [bs, nt, nl], mask of valid tokens
    reduce: if reduce, will return the averaged number
    NOTE: loss_fn should NOT output averaged elements, and should NOT ignore indices. 
    ignored indices should be indicated in masks
    """
    bs, nt, nl, vs = a.shape
    # for each token in a, get the min dist in b  
    token_mask_a, token_mask_b = mask_a.sum(-1) == nl, mask_b.sum(-1) == nl
    num_unmasked_a, num_unmasked_b = nl - mask_a.sum(-1), nl - mask_b.sum(-1) 
    num_unmasked_a = torch.where(num_unmasked_a == 0, nl, num_unmasked_a)
    num_unmasked_b = torch.where(num_unmasked_b == 0, nl, num_unmasked_b)
    _a = a.unsqueeze(2).repeat_interleave(nt, dim=2).permute(0, 1, 4, 2, 3).reshape(bs*nt, vs, nt, nl) # [bs, nt, vs, nt, nl]
    _b = b.unsqueeze(1).repeat_interleave(nt, dim=1).reshape(bs*nt, nt, nl) #[bs, nt, nt, nl]
    a_loss = loss_fn(_a, _b).reshape(bs, nt, nt, nl) # [bs* nt, nt, nl]
    a_loss = a_loss.masked_fill(mask_b.reshape(bs, 1, nt, nl), value=0).sum(-1) / num_unmasked_b.unsqueeze(1)
    a_loss = a_loss.masked_fill(token_mask_b.unsqueeze(1), value=1e9)
    a_loss = a_loss.min(-1)[0] # [bs, nt]
    a_loss = a_loss.masked_fill(token_mask_a, value=0)
    if reduce:
        a_loss = a_loss.sum(-1) / (nt - token_mask_a.sum(-1))
    
    _a = a.unsqueeze(1).repeat_interleave(nt, dim=1).permute(0, 1, 4, 2, 3).reshape(bs*nt, vs, nt, nl) # [bs, nt, vs, nt, nl]
    _b = b.unsqueeze(2).repeat_interleave(nt, dim=2).reshape(bs*nt, nt, nl) #[bs, nt, nt, nl]
    b_loss = loss_fn(_a, _b).reshape(bs, nt, nt, nl) # [bs* nt, nt, nl]
    b_loss = b_loss.masked_fill(mask_a.reshape(bs, 1, nt, nl), value=0).sum(-1) / num_unmasked_a.unsqueeze(1)
    b_loss = b_loss.masked_fill(token_mask_a.unsqueeze(1), value=1e9)
    b_loss = b_loss.min(-1)[0] # [bs, nt]
    b_loss = b_loss.masked_fill(token_mask_b, value=0)
    if reduce:
        b_loss = b_loss.sum(-1) / (nt - token_mask_b.sum(-1))
    
    if split:
        return a_loss.mean(), b_loss.mean()
    loss = a_loss + b_loss 
    if reduce:
        return loss.mean()
    return loss


def minloss(loss_fn: Callable , 
             logits: Shaped[Tensor, "bs sentence_length vs"], 
             labels: Shaped[Tensor, "bs n_perm sentence_length"], 
             reduce: bool=True):
    
    bs, sentence_length, vs = logits.shape
    bs, n_perm, sentence_length = labels.shape

    # loss = loss_fn(
    #             logits.reshape(-1, vs), 
    #             labels[:,0,:].reshape(-1), 
    #             )

    # Expand dimension of logits
    logits_expanded = logits.unsqueeze(1).expand(-1, n_perm, -1, -1)

    # Compute cross entropy loss for each permutation
    loss = loss_fn(logits_expanded.reshape(-1, vs), labels.reshape(-1))

    # Reshape the loss back to (bs, n_perm, sentence_length)
    loss = loss.reshape(bs, n_perm, sentence_length)

    # Take the mean over sentence_length dimension (mean over a sentence)
    loss = loss.mean(dim=-1)

    # Create a mask for all-zero permutations
    mask = (labels.sum(dim=-1) == 0)

    # Use the mask to assign a high value to the all-zero permutations in the loss tensor
    loss[mask] = float('inf')

    print(labels)
    # loss before taking min: 
    # tensor([[4.6327, 4.6373, 4.6422, 4.6026, 4.6505, 4.6377,    inf,    inf,    inf,
    #         inf,    inf,    inf,    inf,    inf,    inf,    inf,    inf,    inf,
    #         inf,    inf,    inf,    inf,    inf,    inf],
    #     [6.6331, 6.6319, 6.6094, 6.6533, 6.6205, 6.6408, 6.6303, 6.6266, 6.6195,
    #      6.6165, 6.6444, 6.6146, 6.6175, 6.6596, 6.6146, 6.6464, 6.6120, 6.6269,
    #      6.6028, 6.6322, 6.6148, 6.6438, 6.6434, 6.6300]],
    #    grad_fn=<IndexPutBackward0>)

    # Take the minimum loss over n_perm dimension
    loss, _ = loss.min(dim=-1)

    # loss after taking min: tensor([4.6026, 6.6028], grad_fn=<MinBackward0>)

    return loss.mean()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecordLoss():
    def __init__(self):
        self.n_iter = 0
        self.losses = AverageMeter()
        self.nlls = AverageMeter()
        self.point_kls = AverageMeter()
        self.logprob_pyxs = AverageMeter()
        self.entropys = AverageMeter()
        self.qx_samples = []
        self.px_samples = []
        self.py_samples = []
        self.measures =(self.losses, self.nlls,self.point_kls,self.logprob_pyxs,self.entropys)

    def reset(self):
        self.__init__()

    def add_samples_qx(self, samples):
        self.qx_samples += [samples]

    def add_samples_py(self, samples):
        self.py_samples += [samples]

    def add_samples_px(self, samples):
        self.px_samples += [samples]

    def reset_samples(self):
        self.qx_samples = []
        self.px_samples = []
        self.py_samples = []

    def update(self, loss=None, nll=None, point_kl=None, logprob_pyx=None, entropy=None):
        for (i,term) in enumerate((loss, nll, point_kl, logprob_pyx, entropy)):
            if term is not None:
                self.measures[i].update(term[0],term[1])
        self.n_iter += 1

def batch_seqs(seqs):
    max_len = max(len(s) for s in seqs)
    data = np.zeros((max_len, len(seqs)))
    for i, s in enumerate(seqs):
            data[:len(s), i] = s
    return torch.LongTensor(data)

def weight_top_p(vec, p):
    indices = (-vec).argsort()
    out = np.zeros_like(vec)
    cumprob = 0
    for i in indices:
        excess = max(0, cumprob + vec[i] - p)
        weight = vec[i] - excess
        out[i] = weight
        cumprob += weight
        if excess > 0:
            break

    out /= out.sum()
    return out

def trim(L, obj):
    if obj in L:
        return L[:L.index(obj)+1]
    return L

"""Noam Scheduler."""



from torch.optim import lr_scheduler
class NoamLR(lr_scheduler._LRScheduler):
    r"""Noam Learning rate schedule.
    Increases the learning rate linearly for the first `warmup_steps` training steps, then decreases it proportional to
    the inverse square root of the step number.
              ^
             / \
            /   `
           /     `
          /         `
         /               `
        /                       `
       /                                   `
      /                                                    `
     /                                                                              `
    /                                                                                                                  `
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser instance to modify the learning rate of.
    warmup_steps : int
        The number of steps to linearly increase the learning rate.
    Notes
    -----
    If step <= warmup_steps,
        scale = step / warmup_steps
    If step > warmup_steps,
        scale = (warmup_steps ^ 0.5) / (step ^ 0.5)
    """
    def __init__(self, optimizer, model_size, warmup_steps=4000):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer)

    def scale(self, step):
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

    def get_lr(self):
        scale = self.scale(max(1,self._step_count))
        return [base_lr * scale for base_lr in self.base_lrs]
