.PHONY: data


data:
	rm -rf data/raw/*
	ln -s /om/group/cpl/language-models/syntaxgym/analysis/full_data.csv data/raw/perplexity.csv
	# Dummy eval output for the moment
	echo -e "model_id,test_suite,item_number,correct\nvanilla_ptb_0111,fgd,1,True" > data/raw/test_suite_results.csv
