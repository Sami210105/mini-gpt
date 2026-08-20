import torch
import torch.nn as nn #building actual neural network layers
import torch.nn.functional as F 

torch.manual_seed(42)

class MultiHeadAttention(nn.Module): #is a nn, inherit from nn.Module -has built in fn --> store parameters, save , load models, move tensors to GPU, register layers and calculate gradients 
    def __init__(self, d_model, n_heads): #embedding size, number of heads
        super().__init__() #call init function from nn.Module (parent class)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads #integer division, size of each head's query/key/value vectors
        
        #These create query, key, and value matrices for each head.
        self.W_q = nn.Linear(d_model, d_model) #nn.Linear computes y = xW^T + b, x=input, W=weight matrix, b=bias vector
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        #This is output projection layer
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, causal=True):
        #x: batch, seq_len, d_model
        batch, seq_len, d_model = x.shape
        
        Q = self.W_q(x) #take every token, multiply by weight add bias and produce query vector
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch, seq_len, self.n_heads, self.d_k)
        K = K.view(batch, seq_len, self.n_heads, self.d_k)
        V = V.view(batch, seq_len, self.n_heads, self.d_k)

        print(f"After view: {Q.shape}")

        #before we have batch, seq_len, heads, d_k, we need to reshape to batch, heads, seq_len, d_k because we want n_heads to have each token and not every token to have n_heads
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        scores = Q @ K.transpose(-2, -1) #to swap seq_len and d_k
        scores = scores / (self.d_k ** 0.5)
        
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        head_outputs = attn_weights @ V
        head_outputs = head_outputs.transpose(1,2) #back to batch, seq_len, heads, d_k
        
        concat = head_outputs.contiguous().view(batch, seq_len, self.d_model)

        output = self.W_o(concat)
        
        return output
    
if __name__ == "__main__":
    batch = 1
    seq_len = 4
    d_model = 8
    n_heads = 2
        
    x = torch.randn(batch, seq_len, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    out = mha(x, causal=True)
    print(f"Running MultiHeadAttention")
        
    assert out.shape == x.shape
    print("Output shape matches input shape!")
    
        
        