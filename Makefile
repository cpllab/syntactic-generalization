.PHONY: data


data:
	rm -rf data/raw/*
	ln -s /om/group/cpl/language-models/syntaxgym/analysis/full_ppl.csv data/raw/perplexity.csv
	ln -s /om/group/cpl/language-models/syntaxgym/analysis/syntaxgym_results.csv data/raw/test_suite_results.csv
