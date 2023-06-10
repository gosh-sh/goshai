import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

def l2norm(t):
    return F.normalize(t, dim = -1)

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True).detach()
    return F.softmax(t, dim = dim)

class KNNMemoryAttention(nn.Module):
    @classmethod
    def is_knn_attention_layer(cls, obj):
        return isinstance(obj, KNNMemoryAttention)

    def __init__(self, num_heads = 8, dropout = 0., num_retrieved_memories = 32, attn_scale_init = 20):
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1) * math.log(attn_scale_init))
        
        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)
        self.knn_mem_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, local_out, knn_memory, add_knn_memory = True):
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)

        # in paper, they showed normalizing of keys led to more stable training
        # we'll just go with full cosine sim attention https://arxiv.org/abs/2010.04245
        # q, k = map(l2norm, (q, k)) # local_out is evaluated with not normalized q and k -> local_out + mem_out will be evaluated with different pipeline

        scale = self.scale.exp()

        mask_value = -torch.finfo(q.dtype).max

        mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim = -2)

        sim_mem = einsum('b h i d, b h i j d -> b h i j', q, mem_k) * scale
        sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)


        if add_knn_memory:
            new_kv_memories = torch.stack((k, v), dim = -2).detach()
            if new_kv_memories.numel() > 0:
                knn_memory.add(new_kv_memories)

        mem_attn = stable_softmax(sim_mem)
        mem_attn = self.dropout(mem_attn)

        mem_out = einsum('b h i j, b h i j d -> b h i d', mem_attn, mem_v)

        mem_out = rearrange(mem_out, 'b h n d -> b n (h d)')
        
        out = local_out + mem_out

        # out = rearrange(out, 'b h n d -> b n (h d)')

        return out

