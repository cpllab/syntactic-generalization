#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import json
import operator
import os
from pathlib import Path
from pprint import pprint
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.notebook import tqdm
tqdm.pandas()
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png')


# ## Load data and preprocess

# ### Metadata

# In[2]:


# Map from test suite tag to high-level circuit.
circuits = {
    "Licensing": ["npi", "reflexive"],
    "Long-Distance Dependencies": ["fgd", "cleft"],
    "Agreement": ["number"],
    "Garden-Path Effects": ["npz", "mvrr"],
    "Gross Syntactic State": ["subordination"],
    "Center Embedding": ["center"],
}

tag_to_circuit = {tag: circuit
                  for circuit, tags in circuits.items()
                  for tag in tags}


# In[3]:


# Exclusions
exclude_suite_re = re.compile(r"^fgd-embed[34]|^gardenpath|^nn-nv")
exclude_models = ["1gram", "ngram", "ngram-no-rand"]


# In[4]:


ngram_models = ["1gram", "ngram", "ngram-single"]
baseline_models = ["random"]

# Models for which we designed a controlled training regime
controlled_models = ["ngram", "ordered-neurons", "vanilla", "rnng"]


# ### Load

# In[5]:


ppl_data_path = Path("../data/raw/perplexity.csv")
test_suite_results_path = Path("../data/raw/test_suite_results")


# In[6]:


perplexity_df = pd.read_csv(ppl_data_path, index_col=["model", "corpus", "seed"])
perplexity_df.index.set_names("model_name", level=0, inplace=True)

results_df = pd.concat([pd.read_csv(f) for f in test_suite_results_path.glob("*.csv")])

# Split model_id into constituent parts
model_ids = results_df.model.str.split("_", expand=True).rename(columns={0: "model_name", 1: "corpus", 2: "seed"})
results_df = pd.concat([results_df, model_ids], axis=1).drop(columns=["model"])
results_df["seed"] = results_df.seed.fillna("0").astype(int)

# Exclude test suites
exclude_filter = results_df.suite.str.contains(exclude_suite_re)
print("Dropping %i results / %i suites due to exclusions:"
      % (exclude_filter.sum(), len(results_df[exclude_filter].suite.unique())))
print(" ".join(results_df[exclude_filter].suite.unique()))
results_df = results_df[~exclude_filter]

# Exclude models
exclude_filter = results_df.model_name.isin(exclude_models)
print("Dropping %i results due to dropping models:" % exclude_filter.sum(), list(results_df[exclude_filter].model_name.unique()))
results_df = results_df[~exclude_filter]

# Add tags
results_df["tag"] = results_df.suite.transform(lambda s: re.split(r"[-_0-9]", s)[0])
results_df["circuit"] = results_df.tag.map(tag_to_circuit)
tags_missing_circuit = set(results_df.tag.unique()) - set(tag_to_circuit.keys())
if tags_missing_circuit:
    print("Tags missing circuit: ", ", ".join(tags_missing_circuit))


# In[7]:


results_df.head()


# In[8]:


# Average across seeds of each ngram model.
# The only difference between "seeds" of these model types are random differences in tie-breaking decisions.
for ngram_model in ngram_models:
    # Create a synthetic results_df with one ngram model, where each item is correct if more than half of
    # the ngram seeds vote.
    ngram_results_df = (results_df[results_df.model_name == ngram_model].copy().groupby(["model_name", "corpus", "suite", "item", "tag", "circuit"]).agg({"correct": "mean"}) > 0.5).reset_index()
    ngram_results_df["seed"] = 0
    
    # Drop existing model results.
    results_df = pd.concat([results_df[~(results_df.model_name == ngram_model)],
                            ngram_results_df], sort=True)


# ### Data prep

# In[9]:


suites_df = results_df.groupby(["model_name", "corpus", "seed", "suite"]).correct.mean().reset_index()
suites_df["tag"] = suites_df.suite.transform(lambda s: re.split(r"[-_0-9]", s)[0])
suites_df["circuit"] = suites_df.tag.map(tag_to_circuit)

# For controlled evaluation:
# Compute a model's test suite accuracy relative to the mean accuracy on this test suite.
# Only compute this on controlled models.
def get_controlled_mean(suite_results):
    # When computing test suite mean, first collapse test suite accuracies within model--corpus, then combine resulting means.
    return suite_results[suite_results.model_name.isin(controlled_models)].groupby(["model_name", "corpus"]).correct.mean().mean()
suite_means = suites_df.groupby("suite").apply(get_controlled_mean)
suites_df["correct_delta"] = suites_df.apply(lambda r: r.correct - suite_means.loc[r.suite] if r.model_name in controlled_models else None, axis=1)


# In[10]:


# Join PPL and accuracy data.
joined_data = suites_df.groupby(["model_name", "corpus", "seed"])[["correct", "correct_delta"]].agg("mean")
joined_data = pd.DataFrame(joined_data).join(perplexity_df).reset_index()
joined_data.head()


# In[11]:


# Join PPL and accuracy data, splitting on circuit.
joined_data_circuits = suites_df.groupby(["model_name", "corpus", "seed", "circuit"])[["correct", "correct_delta"]].agg("mean")
joined_data_circuits = pd.DataFrame(joined_data_circuits).reset_index().set_index(["model_name", "corpus", "seed"]).join(perplexity_df).reset_index()
joined_data_circuits.head()


# In[12]:


# Analyze stability to modification.
def has_modifier(ts):
    if ts.endswith(("_modifier", "_mod")):
        return True
    else:
        return None
suites_df["has_modifier"] = suites_df.suite.transform(has_modifier)

# Mark "non-modifier" test suites
modifier_ts = suites_df[suites_df.has_modifier == True].suite.unique()
no_modifier_ts = [re.sub(r"_mod(ifier)?$", "", ts) for ts in modifier_ts]
suites_df.loc[suites_df.suite.isin(no_modifier_ts), "has_modifier"] = False
# Store subset of test suites which have definite modifier/no-modifier marking
suites_df_mod = suites_df[~(suites_df.has_modifier.isna())].copy()
# Get base test suite (without modifier/no-modifier marking)
suites_df_mod["test_suite_base"] = suites_df_mod.suite.transform(lambda ts: ts.strip("_no-modifier").strip("_modifier"))
suites_df_mod.head()


# ### Checks

# In[13]:


# Each model--corpus--seed should have perplexity data.
ids_from_results = results_df.set_index(["model_name", "corpus", "seed"]).sort_index().index
ids_from_ppl = perplexity_df.sort_index().index
diff = set(ids_from_results) - set(ids_from_ppl)
if diff:
    print("Missing perplexity results for:")
    pprint(diff)
    #raise ValueError("Each model--corpus--seed must have perplexity data.")


# In[14]:


# Every model--corpus--seed should have results for all test suite items.
item_list = {model_key: set(results.suite)
             for model_key, results in results_df.groupby(["model_name", "corpus", "seed"])}
not_shared = set()
for k1, k2 in itertools.combinations(item_list.keys(), 2):
    l1, l2 = item_list[k1], item_list[k2]
    if l1 != l2:
        print("SyntaxGym test suite results for %s and %s don't match" % (k1, k2))
        print("\tIn %s but not in %s:\n\t\t%s" % (k2, k1, l2 - l1))
        print("\tIn %s but not in %s:\n\t\t%s" % (k1, k2, l1 - l2))
        print()
        
        not_shared |= l2 - l1
        not_shared |= l1 - l2

if len(not_shared) > 0:
    to_drop = results_df[results_df.suite.isin(not_shared)]
    print("Dropping these test suites (%i rows) for now. Yikes:" % len(to_drop))
    print(not_shared)
    results_df = results_df[~results_df.suite.isin(not_shared)]
else:
    print("OK")


# In[15]:


# Second sanity check: same number of results per model--corpus--seed
result_counts = results_df.groupby(["model_name", "corpus", "seed"]).item.count()
if len(result_counts.unique()) > 1:
    print("WARNING: Some model--corpus--seed combinations have more result rows in results_df than others.")
    print(result_counts)


# In[16]:


# Second sanity check: same number of suite-level results per model--corpus--seed
suite_result_counts = suites_df.groupby(["model_name", "corpus", "seed"]).suite.count()
if len(suite_result_counts.unique()) > 1:
    print("WARNING: Some model--corpus--seed combinations have more result rows in suites_df than others.")
    print(suite_result_counts)


# ## Prepare for data rendering

# In[17]:


RENDER_FINAL = True
figure_path = Path("../reports/figures")
figure_path.mkdir(exist_ok=True, parents=True)

RENDER_CONTEXT = {
    "font_scale": 3.5,
    "rc": {"lines.linewidth": 2.5}
}


# In[18]:


# Establish consistent orderings of model names, corpus names, circuit names
# for figure ordering / coloring
model_order = sorted(set(results_df.model_name))
controlled_model_order = sorted(set(results_df.model_name) & set(controlled_models))
corpus_order = ["bllip-lg", "bllip-md", "bllip-sm", "bllip-xs"]
circuit_order = sorted([c for c in results_df.circuit.dropna().unique()])


# ## Main analyses

# ### Baseline sanity check

# In[19]:


baselines_to_plot = set(baseline_models) & set(suites_df.model_name.unique())
f, axs = plt.subplots(len(baselines_to_plot), 1, figsize=(40, 10 * len(baselines_to_plot)))

for baseline_model, ax in zip(baselines_to_plot, np.ravel(axs)):
    sns.barplot(data=suites_df[suites_df.model_name == baseline_model], x="suite", y="correct", ax=ax)
    plt.title(f"{baseline_model} test suite results, averaged across corpus size")


# ### Basic barplots

# In[20]:


plt.subplots(figsize=(10, 7))
sns.barplot(data=suites_df.reset_index(), x="model_name", y="correct")

plt.xlabel("Model")
plt.ylabel("Accuracy")


# In[21]:


plt.subplots(figsize=(20, 10))
sns.barplot(data=joined_data_circuits, x="circuit", y="correct", hue="model_name", hue_order=model_order)


# ### Controlled evaluation of model type + dataset size

# In[22]:


controlled_suites_df = suites_df[suites_df.model_name.isin(controlled_models)]
controlled_suites_df_mod = suites_df_mod[suites_df_mod.model_name.isin(controlled_models)]
controlled_joined_data_circuits = joined_data_circuits[joined_data_circuits.model_name.isin(controlled_models)]


# In[23]:


# Compare SG deltas w.r.t. test suite mean rather than absolute values.
# This makes for a more easily interpretable visualization

f, ax = plt.subplots(figsize=(15, 10))
ax.axhline(0, c="gray", alpha=0.5, linestyle="--")
sns.barplot(data=controlled_suites_df.reset_index(), x="model_name", y="correct_delta", order=controlled_model_order, ax=ax)
sns.swarmplot(data=controlled_suites_df.reset_index(), x="model_name", y="correct_delta", order=controlled_model_order, alpha=0.7, ax=ax)

plt.xlabel("Model")
plt.ylabel("Delta from per-suite mean accuracy")
plt.title("Model averages: delta from mean accuracy")


# In[24]:


plt.subplots(figsize=(15, 10))
# Estimate error intervals with a structured bootstrap: resampling units = model
sns.barplot(data=controlled_suites_df.reset_index(), x="corpus", y="correct_delta",
            units="model_name", order=corpus_order)

plt.xlabel("Corpus")
plt.ylabel("Delta from per-suite mean accuracy")
plt.title("Corpus averages: delta from mean accuracy")


# In[25]:


plt.subplots(figsize=(20, 10))
sns.barplot(data=controlled_joined_data_circuits, x="circuit", y="correct_delta", hue="model_name", hue_order=controlled_model_order)

# TODO swarmplot split across corpus


# In[26]:


plt.subplots(figsize=(20, 10))
sns.barplot(data=controlled_joined_data_circuits, x="circuit", y="correct_delta", hue="corpus", units="model_name", hue_order=corpus_order)


# #### Stability to modification

# In[27]:


controlled_suites_df_mod.suite.unique()


# In[28]:


plt.subplots(figsize=(15, 10))
sns.barplot(data=controlled_suites_df_mod, x="model_name", y="correct", hue="has_modifier", order=controlled_model_order)
plt.title("Stability to modification")


# In[29]:


plt.subplots(figsize=(15, 10))
sns.barplot(data=controlled_suites_df_mod, x="corpus", y="correct", hue="has_modifier", units="model_name", order=corpus_order)
plt.title("Stability to modification")


# In[30]:


g = sns.FacetGrid(data=controlled_suites_df_mod, col="model_name", height=7)
g.map(sns.barplot, "corpus", "correct", "has_modifier", order=corpus_order, hue_order=[False, True])
g.add_legend()


# In[31]:


avg_mod_results = controlled_suites_df_mod.groupby(["model_name", "test_suite_base", "has_modifier"]).correct.agg(correct="mean").sort_index()
avg_mod_diffs = avg_mod_results.xs(True, level="has_modifier") - avg_mod_results.xs(False, level="has_modifier")

plt.subplots(figsize=(15, 10))
sns.boxplot(data=avg_mod_diffs.reset_index(), x="model_name", y="correct", order=controlled_model_order)
plt.title("Change in accuracy due to modification")


# ### Accuracy vs perplexity

# In[32]:


f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=joined_data, x="test_ppl", y="correct",
                hue="model_name", style="corpus", s=150,
                hue_order=model_order)
plt.xlabel("Test corpus perplexity")
plt.ylabel("SyntaxGym absolute scores vs. perplexity")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Add horizontal lines for models without ppl estimates.
no_ppl_data = joined_data[joined_data.test_ppl.isna()]
for model_name, rows in no_ppl_data.groupby("model_name"):
    y = rows.correct.mean()
    # TODO match legend color
    # TODO show error region?
    ax.axhline(y, linestyle="dashed")
    ax.text(110, y + 0.0025, model_name, alpha=0.7)


# In[33]:


f, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=joined_data[joined_data.model_name.isin(controlled_models)], x="test_ppl", y="correct_delta",
                hue="model_name", style="corpus", s=150,
                hue_order=controlled_model_order)
plt.xlabel("Test corpus perplexity")
plt.ylabel("SyntaxGym delta score")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title("SyntaxGym delta scores vs. perplexity")


# In[34]:


g = sns.lmplot(data=joined_data, x="test_ppl", y="correct_delta",
               hue="corpus", hue_order=corpus_order, truncate=True)
g.ax.set_ylim((joined_data.correct_delta.min() - 0.1, joined_data.correct_delta.max() + 0.1))


# In[35]:


g = sns.lmplot(data=joined_data[joined_data.model_name.isin(controlled_models)], x="test_ppl", y="correct_delta",
               hue="model_name", hue_order=controlled_model_order, truncate=True)
g.ax.set_ylim((joined_data.correct_delta.min() - 0.1, joined_data.correct_delta.max() + 0.1))


# In[36]:


g = sns.FacetGrid(data=joined_data_circuits, col="circuit", height=5)
g.map(sns.scatterplot, "test_ppl", "correct", "model_name",
      hue_order=model_order, s=100)
g.add_legend()


# In[37]:


g = sns.FacetGrid(data=controlled_joined_data_circuits, col="circuit", row="model_name", height=5)
g.map(sns.scatterplot, "test_ppl", "correct",
      hue_order=controlled_model_order, s=100)
g.add_legend()


# ### Item-level statistics

# In[38]:


EXCLUDE_FROM_ITEM_ANALYSIS = ["random"]


# #### Item-level prediction correlations across models

# In[39]:


item_predictions = results_df[~results_df.model_name.isin(EXCLUDE_FROM_ITEM_ANALYSIS)]     .set_index(["suite", "item"]).sort_index().groupby(["model_name", "corpus", "seed"]).correct.apply(np.array)
model_correlations, model_agreement = [], []
for k1, k2 in itertools.combinations(list(item_predictions.index), 2):
    k1_key = " ".join(map(str, k1))
    k2_key = " ".join(map(str, k2))
    k1_vals = item_predictions.loc[k1]
    k2_vals = item_predictions.loc[k2]
    
    model_correlations.append((k1_key, *k1, k2_key, *k2, stats.spearmanr(k1_vals, k2_vals)[0]))
    model_agreement.append((k1_key, *k1, k2_key, *k2, (k1_vals == k2_vals).mean()))

corr_df = pd.DataFrame(model_correlations, columns=["key_1", "model_1", "corpus_1", "seed_1", "key_2", "model_2", "corpus_2", "seed_2", "corr"])
agree_df = pd.DataFrame(model_agreement, columns=["key_1", "model_1", "corpus_1", "seed_1", "key_2", "model_2", "corpus_2", "seed_2", "agreement"])


# In[40]:


plt.subplots(figsize=(10, 10))
sns.heatmap(data=corr_df.pivot("key_1", "key_2", "corr"))


# In[41]:


plt.subplots(figsize=(10, 10))
sns.heatmap(data=agree_df.pivot("key_1", "key_2", "agreement"))


# In[42]:


plt.subplots(figsize=(10, 10))
sns.distplot(results_df[~results_df.model_name.isin(EXCLUDE_FROM_ITEM_ANALYSIS)].groupby(["suite", "item"]).correct.agg("mean"), bins=20)
plt.title("Distribution of item-level accuracy means")


# In[43]:


plt.subplots(figsize=(10, 10))
sns.distplot(results_df[~results_df.model_name.isin(EXCLUDE_FROM_ITEM_ANALYSIS)].groupby(["suite", "item"]).correct.agg("std"), bins=20)
plt.title("Distribution of item-level accuracy stdevs")


# In[44]:


# Get items for which all models fail / succeed
all_fail = results_df[~results_df.model_name.isin(EXCLUDE_FROM_ITEM_ANALYSIS) & results_df.model_name.isin(controlled_models)].groupby(["suite", "item"]).correct.max() == False
all_succeed = results_df[~results_df.model_name.isin(EXCLUDE_FROM_ITEM_ANALYSIS) & results_df.model_name.isin(controlled_models)].groupby(["suite", "item"]).correct.min() == True

# Get items for which each condition is true
all_fail = all_fail[all_fail]
all_succeed = all_succeed[all_succeed]

print("All fail\n", all_fail)
print("All succeed\n", all_succeed)


# In[45]:


plt.subplots(figsize=(10, 10))
sns.distplot(suites_df[~suites_df.model_name.isin(EXCLUDE_FROM_ITEM_ANALYSIS)].correct, bins=20)
plt.title("Distribution of suite-level accuracy means")


# ## Circuit–circuit correlations

# In[46]:


# Exclude some models from circuit correlation analysis.
EXCLUDE_FROM_CIRCUIT_ANALYSIS = ["random", "ngram", "1gram", "ngram-single"]


# In[47]:


f, axs = plt.subplots(len(circuit_order), len(circuit_order), figsize=(25, 25))
plt.subplots_adjust(hspace=0.6, wspace=0.6)

source_df = suites_df[~suites_df.model_name.isin(EXCLUDE_FROM_CIRCUIT_ANALYSIS)]

for c1, row in zip(circuit_order, axs):
    for c2, ax in zip(circuit_order, row):
        if c1 <= c2:
            ax.axis("off")
            continue
            
        xs = source_df[source_df.circuit == c1].groupby(["model_name", "corpus", "seed"]).correct.agg({c1: "mean"})
        ys = source_df[source_df.circuit == c2].groupby(["model_name", "corpus", "seed"]).correct.agg({c2: "mean"})
        df = pd.concat([xs, ys], axis=1)
        ax.set_title("%s /\n %s" % (c1, c2))
        sns.regplot(data=df, x=c1, y=c2, ax=ax)
        
plt.suptitle("Circuit--circuit correlations")


# In[48]:


# Estimate lower-bound Spearman r for each circuit-circuit relation
# by running a structured bootstrap over model--corpus--seeds: randomly
# resample model--corpus--seed combinations and recompute Spearman r's.
def estimate_r(xs):
    # Calculate Spearman-r on bootstrap sample comparing two circuits (shape n * 2)
    corr, pval = stats.spearmanr(xs[:, 0], xs[:, 1])
    return corr

corr_data = pd.DataFrame(index=circuit_order, columns=circuit_order)
n_boot = 500
for c1, c2 in tqdm(list(itertools.combinations(circuit_order, 2))):
    xs = source_df[source_df.circuit == c1].groupby(["model_name", "corpus", "seed"]).correct.agg({c1: "mean"})
    ys = source_df[source_df.circuit == c2].groupby(["model_name", "corpus", "seed"]).correct.agg({c2: "mean"})

    df = pd.concat([xs, ys], axis=1)
    # Concatenate model--corpus--seed labels to make structured bootstrapping easier.
    df["model_key"] = [" ".join(map(str, key)) for key in df.index.tolist()]
    df = df.reset_index(drop=True)

    corr_data.loc[c1, c2] = sns.utils.ci(sns.algorithms.bootstrap(df, units=df.model_key, n_boot=n_boot, func=estimate_r))


# In[49]:


corr_data


# ## Quantitative tests
# 
# `SG ~ ppl:corpus + model_name + (1 | test_suite)`

# In[53]:


perplexity_map = perplexity_df.test_ppl.to_dict()

reg_df = results_df[results_df.model_name.isin(controlled_models)].copy()
reg_df["test_ppl"] = pd.Series(list(zip(reg_df.model_name, reg_df.corpus, reg_df.seed))).map(perplexity_map)
reg_df.to_csv("reg_df.csv")

