import math
from torch.nn import functional as F
import torch.nn as nn
import torch
import inspect




def attention(q, k, v):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k要转置（交换最后两个维度）
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def example_attention():
    # query: fruit 
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (1, 4)
    
    # keys
    keys = torch.tensor([[1.0, 0.5, 0.0, 0.0],  # apple - 强匹配
                         [0.9, 0.4, 0.0, 0.0],  # banana - 强匹配
                         [0.0, 0.0, 0.8, 1.0],  # cat - 不匹配
                         [0.0, 0.0, 0.9, 0.8]])  # dog - 不匹配
    
    # values: 实际的语义内容
    values = torch.tensor([[1.0, 0.0, 0.5, 0.0],  # apple - 红色，圆形
                           [0.0, 1.0, 0.0, 0.8],  # banana - 黄色，长形
                           [0.5, 0.5, 0.0, 0.0],  # cat - 灰色，动物
                           [0.3, 0.2, 0.0, 0.0]])  # dog - 棕色，动物
    
    output = attention(query, keys, values)
    print(output) #输出的就是水果相关的属性信息 应该更接近苹果和香蕉的属性


example_attention()

class MultiHeadAttention(nn.Module):
    def __init__(self, config, is_causal=False):
        # config包含模型的配置参数，这里可以先不管
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attns = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=config.bias) for _ in range(3)]) # 原文备注：矩阵内积再拼接等价于拼接后再内积
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = is_causal


    def forward(self, query, key, value):

        B, T, C = query.size() # batch size, sequence length, embedding dimensionality (n_embd)
  
        q, k, v  = [self.c_attns[i](x) for i, x in zip(range(3), (query, key, value))]

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.is_causal:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # 第一个线性层
        self.relu = nn.ReLU()  # 加一个relu
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) # 第二个线性层
        self.dropout = nn.Dropout(config.dropout) 
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5) # torch自带的layer_norm函数 我们加上了bias
    

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config, is_causal=False)  # 前后各一个LayerNorm
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # 此处前面加了 x 实则是实现了残差连接
        x = self.ln_1(x)
        # Encoder 使用 Self Attention，所以 Q、K、V 都是 x
        # print("x",x.size())
        x = x + self.attn(x, x, x)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__() 
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)]) # 这里要用多层EncoderLayer
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias) 
        self.m_attn = MultiHeadAttention(config, is_causal=True) # decoder这里的att是需要加掩码的（只能看到前面）
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = MultiHeadAttention(config, is_causal=False) # 这里不加掩码 因为是和encoder的输出做attention
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, enc_out):
        x = self.ln_1(x)
        x = x + self.m_attn(x, x, x) # 自注意力
        x = self.ln_2(x)

        x = x + self.attn(x, enc_out, enc_out)
        x = self.ln_3(x)
        x = x + self.mlp(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__() 
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)]) # 这里要用多层DecoderLayer
        self.norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)

class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)

        pe = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2) * -(math.log(10000.0) / config.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    



class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 必须输入词表大小和 block size
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = PositionalEncoding(config),
            drop = nn.Dropout(config.dropout),
            encoder = Encoder(config),
            decoder = Decoder(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)


    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.config.block_size}"


        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(tok_emb) 
        x = self.transformer.drop(pos_emb)
        enc_out = self.transformer.encoder(x)
        x = self.transformer.decoder(x, enc_out)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]


        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)


        return optimizer


    @torch.no_grad() # 不需要计算梯度
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:] #做截断
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

# 直接用torch封装好的Transformer模块是下面这样写
"""
class Transformer_Torch(nn.Module):
    def __init__(self, config):
        super(Transformer_Torch, self).__init__()
        self.model = nn.Transformer(d_model=config.n_embd,
                                    nhead=config.n_head,
                                    num_encoder_layers=config.n_layer,
                                    num_decoder_layers=config.n_layer,
                                    dim_feedforward=4*config.n_embd,
                                    dropout=config.dropout,
                                    activation='relu',
                                    custom_encoder=None,
                                    custom_decoder=None)
    
    def forward(self, src, tgt):
        output = self.model(src, tgt)
        return output
"""