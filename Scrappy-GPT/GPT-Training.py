import torch
import torch.nn as nn
from torch.nn import functional as F

# Define important variables (HYPER PARAMETERS)
batch_size = 64
block_size = 256
max_iters = 6000
eval_interval = 500
learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#-----------------
torch.manual_seed(1337)

# Open the Shakespeare file
with open('Shake-the-speare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings from int to char and reverse
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder and decoder functions
encode = lambda s: [stoi[c] for c in s]  # Encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: take a list of integers, output a string

#train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# lets create the head module for self-attention
class Head(nn.Module):
    """One head of self attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Note: tril is not a parameter of the module( in terms of pytorch naming conventions this is a buffer)
        # you have to assign tril as a variable to the model using a register buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, 16)
        q = self.query(x) #(B, T, 16)

        # now for the communication ( computing attention scores >> affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # tokens from the past cant communicate ( but maybe sometimes we want to so we can delete this matrix (in the case of figuring out sentience))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # performing weighted agregation of the values
        v = self.value(x) #(B, T, C)
        out = wei @ v # #(B, T, T) @ (B, T, C) >>>> (B, T, C)
        return out    


#soo single head just made our loss too depressed (it fell too quick)
# Lets implement multiple heads to do this stuff in parallel

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        #note: the projection is the linear transformation of the outcome of this layer
        out = self.dropout(self.proj(out)) 
        return out
    


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'll like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#defining out bigram model (lecture one for final lecture talk about taking things full circle kaparthy)
class BigramLanguageModel(nn.Module):

    def __init__(self):#note: we removed vocab size here cause its a global var
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # changed vocab size to n_embd cause we wanna go thru an intermediate phase before embedding the logits
        #second position embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #creating the self attention head
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) #ie 4 heads of 8-dim self-attention
        #feed forward
        #self.ffwd = FeedForward(n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        #linear layer to go from tok_emb to n_embd 
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    
    def forward(self, idx, targets=None):
        B, T = idx.shape        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) batch, time and channel tensor. batch is 4, time is 8(block size), channel = vocab_size = 65
        #token embeddings ^^ 
        #pos embedding which is the positional embedding int from 0 to T-1
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        # x = self.sa_heads(x) # applying one head of self attention
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # reshaping our logits cause pytorch likes (B,C,T)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # remember corss entropy is the neg log liklihood loss

        return logits, loss

    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block size tokens
            idx_cond = idx[:, -block_size:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel() #note: we removed vocab size here cause its a global var
m = model.to(device)

#lets create the pytorch optimizer
# create a PyTorch optimization object
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) #in makemore we used stocastic gradient descent (SGD) but here we use ADAM.

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch 
    xb, yb = get_batch('train')
    
    #evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000) [0].tolist()))