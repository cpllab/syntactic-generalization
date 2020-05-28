import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

import data
import model

from utils import batchify
from get_raw import unkify


parser = argparse.ArgumentParser()
parser.add_argument("model_checkpoint")
parser.add_argument("file", type=argparse.FileType("r"))
parser.add_argument("--outf", type=argparse.FileType("w"), default=sys.stdout)
parser.add_argument("--corpus_file", type=Path, required=True, help="saved Corpus object from training run")
parser.add_argument("--bptt", type=int, default=70, help="sequence length")
parser.add_argument("--emsize", type=int, default=400, help="size of word embeddings")
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", dest="cuda", action="store_true")
parser.add_argument("--no-cuda", dest="cuda", action="store_false")
parser.set_defaults(cuda=True)

args = parser.parse_args()

# Set the random seed manually for reproducibility.
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        kwargs = {}
        if not args.cuda:
            kwargs["map_location"] = "cpu"
        model, criterion, optimizer = torch.load(f, **kwargs)


def unkify_token(token, vocabulary):
    return unkify([token], vocabulary)[0]


def get_batch(data_source, i, window):
    seq_len = min(window, len(data_source) - 1 - i)
    data = data_source[i:i + seq_len]
    target = data_source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def get_surprisals(sentences, corpus, outf, seed):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    vocabulary = set(corpus.dictionary.word2idx)

    outf.write("sentence_id\ttoken_id\ttoken\tsurprisal\n")

    for i, sentence in enumerate(sentences):
        set_seed(seed)
        outf.write("%i\t%i\t%s\t%f\n" % (i + 1, 1, sentence[0], 0.0))

        sentence = sentence
        data_source = torch.LongTensor(len(sentence))
        for j, token in enumerate(sentence):
            if token not in vocabulary:
                token = unkify_token(token, vocabulary)

            data_source[j] = corpus.dictionary.word2idx[token]

        # model expects T * batch_size array
        data_source = data_source.unsqueeze(1)

        with torch.no_grad():
            hidden = model.init_hidden(1)
            for j in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, j, args.bptt)
                output, hidden = model(data, hidden)
                logprobs = torch.nn.functional.log_softmax(
                        torch.nn.functional.linear(output, model.decoder.weight, bias=model.decoder.bias),
                        dim=1)

                # Convert to numpy and change to log2.
                logprobs = logprobs.detach().numpy()
                logprobs /= np.log(2)

                # Retrieve relevant surprisal values.
                targets = targets.numpy()
                target_surprisals = -logprobs[np.arange(len(targets)), targets]

                for k, surp in enumerate(target_surprisals):
                    outf.write("%i\t%i\t%s\t%f\n" % (i + 1, j + k + 2, sentence[j + k + 1], surp))


corpus = torch.load(args.corpus_file)
model_load(args.model_checkpoint)
sentences = [line.strip().split(" ") for line in args.file.readlines() if line.strip()]
get_surprisals(sentences, corpus, args.outf, args.seed)
