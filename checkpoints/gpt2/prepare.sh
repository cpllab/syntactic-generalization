#!/usr/bin/env bash

jq -r 'keys | .[]' encoder.json > vocab
