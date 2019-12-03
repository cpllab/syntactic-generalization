.PHONY: data


data:
	rm -rf data/raw/*
	mkdir -p data/raw/test_suite_results
	ln -s /om/group/cpl/language-models/syntaxgym/analysis/full_ppl.csv data/raw/perplexity.csv
	ln -s /om/group/cpl/language-models/syntaxgym/analysis/syntaxgym_results_*.csv data/raw/test_suite_results
