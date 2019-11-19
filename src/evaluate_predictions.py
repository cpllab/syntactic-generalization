"""
Evaluate a directory of SyntaxGym result JSON files.

Outputs a CSV describing prediction performance for each model, test suite, and
test suite item.
"""

from argparse import ArgumentParser
from io import StringIO
import json
import regex as re
import numpy as np
import sys
from os import listdir
import pandas as pd

DUMMY_RESULTS_PREFIX = "./dummy_data_results/"

def get_region_results(reg_cond, item):
    reg_cond = eval(reg_cond)
    d = [x['regions'][reg_cond[0]-1]["metric_value"]['sum'] for x in item["conditions"] if x["condition_name"] == reg_cond[1]]
    if (len(d) > 1):
        raise Exception('Two conditions have the same name')
    if (len(d) < 1):
        raise Exception('No matching conditions / regions found')
    else:
        return str(d[0])


def evaluate_prediction(pred, items):
    """
    Evaluate an arbitrary prediction expression.
    """
    result = []
    for itm in items:
        x = re.sub(r'\((.+?)\)', lambda x: get_region_results(x.group(), itm), pred)
        x = x.replace("[", "(").replace("]", ")")
        x = x.replace("=", "==")
        try:
            result.append(eval(x))
        except:
            print("Malformed prediction format (check bracketing)")
    return result


def evaluate_test_suite(to_eval):

  performance_results = []

  for suite in to_eval:

    with open(DUMMY_RESULTS_PREFIX + suite, "r") as inf:
        results = json.load(inf)

    model = results["meta"]["name"]
    preds = results["meta"]["string_predictions"]
    preds = [x.replace(";",",").replace("%","'") for x in preds]

    pred_results = []
    for pred in preds:
        result = evaluate_prediction(pred, results["items"])
        pred_results.append(result)

    pred_results = pred_results[0] #Only getting the first prediction for now
    pred_results = [(model, i, pred_results[i]) for i in range(len(pred_results))]
    performance_results.extend(pred_results)

  df = pd.DataFrame(performance_results)
  return df


def main(test_suite_name):
  if not test_suite_name:
    to_jsonify = [x for x in listdir(DUMMY_RESULTS_PREFIX) if not x.startswith(".")]
  else:
    to_jsonify = test_suite_name

  df = evaluate_test_suite(to_jsonify)
  output = StringIO()
  df.to_csv(output, index=False)

  print(output.get_value())


if __name__ == '__main__':
  main(sys.argv[1:])
