#!/usr/bin/env bash
# Run within the ngram container, from any checkpoint dir mounted at PWD.

/opt/srilm/bin/i686-m64/get-unigram-probs model.lm | cut -d" " -f1 > vocab
