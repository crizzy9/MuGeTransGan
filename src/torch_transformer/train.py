import copy, time
import torch.nn as nn
from transformer import *
from modules import *


def create_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Constructs the transformer model
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

    # Initialize parameters with Glorot / fan_avg.
    # ???
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_epoch(iters, model, loss_fn):
    """
    Start training and log output
    """
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    print_every = 50

    print(iters)
    
    for i, batch in enumerate(iters):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_fn(out, batch.tgt_y, batch.n_tokens) #!!!
        print("i", i)
        print("here", loss)
        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens
        print("batch tokens", batch.n_tokens)
        if i % print_every == 1:
            elapsed = time.time() - start_time
            print("Epoch Step: {}, Loss: {}, Tokens/Second: {}, Time Taken: {}".format(i, loss/batch.n_tokens, tokens/elapsed, elapsed))
            start_time = time.time()
            tokens = 0

    print("tot loss", total_loss)
    print("tot  tokens", total_tokens)
    return total_loss / total_tokens



if __name__  == '__main__':
    # tmp_model = create_model(10, 10, 2)
    # print(tmp_model)

    # Train the simple copy task.
    V = 11
    # random_data = list(random_data_gen(V, 30, 20))[0].src
    # print(random_data)
    # random_data2 = list(random_data_gen(V, 1, 20))[0].src
    # print(random_data2)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = create_model(V, V, N=2)
    # print(model.src_embed(random_data)[0])
    # print(model)
    model_opt = DecayingOptimizer(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        print("epoch {}".format(epoch))
        model.train()
        try:
            print("reached  here")
            run_epoch(random_data_gen(V, 1, 20), model, 
                      SimpleLossCompute(model.generator, criterion, model_opt))
            model.eval()
            print(run_epoch(random_data_gen(V, 1, 5), model, 
                            SimpleLossCompute(model.generator, criterion, None)))
        except Exception as e:
            print(e)
            raise e


