# Test suites

The test suites used to evaluate our models are currently released in `.json` and `.txt` format (`.csv` coming soon!). For a full description of each test suite, please see Appendix B of [our ACL 2020 paper](https://arxiv.org/abs/2005.03692).

The [txt](txt) folder formats each test suite as a `.txt` file with one sentence on each line. There is no final punctuation (unless necessary for the test suite), and the content is not tokenized or unkified. 

The [json](json) folder formats each test suite as a `.json` file containing meta information such as predictions, regions, and experimental conditions in addition to the text content. (Please disregard the numbers in the `sum` field of the `metric_value` sub-dictionary for each region.)
