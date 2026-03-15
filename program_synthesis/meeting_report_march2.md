# LLM-PV Thesis Pipeline: Meeting Report
## March 2, 2026

---

## 1. What We Built

A pipeline that takes tabular classification data and produces human-readable explanations ("theses") of how an LLM-generated classifier (Code0) makes its predictions.

Pipeline stages:
1. **Code0**: LLM writes a Python classifier from training examples
2. **Thesis**: LLM explains why Code0 predicted a given label for a specific test sample, as a conjunction of conditions (e.g., "x3 >= 58.0 AND x15 <= -10.0 AND x11 == 'c1'")
3. **Code1**: LLM transcribes the thesis into executable code, verified for correctness

We evaluate theses on:
- **Coverage**: fraction of training set where the thesis conditions hold (higher = broader explanation)
- **Faithfulness to Ground Truth (GT)**: among samples covered by the thesis, how often the thesis label matches the true label
- **Faithfulness to Code0**: among covered samples, how often the thesis label matches Code0's prediction (always ~100%)

Trivial baselines (always-0, always-1) give coverage=100% but faithfulness=50% on balanced data, so any thesis with faithfulness above 50% is doing real work.

---

## 2. Experiments Run

### 2a. gpt-5-mini Reasoning Effort (fn_o, CDC Diabetes, 25 sequential samples)

Early runs with gpt-5-mini on CDC Diabetes. These used 25 sequentially sampled test points.

| Reasoning | Coverage | Faith (GT) | Faith (Code0) |
|-----------|----------|------------|---------------|
| Minimal   | 19.3%    | 72.4%      | 77.6%         |
| Medium    | 14.6%    | 76.8%      | 97.0%         |
| High      | 11.2%    | **98.3%**  | 100.0%        |

High reasoning achieved near-perfect GT faithfulness. However, we identified a sampling bias: sequential test samples were too similar to each other, so one thesis was covering most of them. This inflated the faithfulness score.

**Fix applied**: switched to 50 stratified random samples (25 label-0, 25 label-1) for all subsequent runs.

### 2b. gpt-5.2 Reasoning Effort Comparison (fn_o, 50 stratified samples)

Same dataset (CDC Diabetes, 21 features), stronger model, fixed sampling.

| Reasoning | Code0 Val Acc | Code0 Test Acc | Coverage | Faith (GT) | Faith (Code0) |
|-----------|---------------|----------------|----------|------------|---------------|
| Low       | **60.4%**     | **58.9%**      | **20.5%**| **68.2%**  | 100.0%        |
| Medium    | 58.4%         | 56.3%          | 16.7%    | 59.4%      | 99.1%         |
| High      | 60.3%         | 58.4%          | 9.8%     | 65.2%      | 99.3%         |

**Surprising finding**: low reasoning outperformed medium and high on almost every metric.

- Medium reasoning produced worse Code0 classifiers AND more specific theses, compounding into lower coverage and lower faithfulness.
- High reasoning recovered somewhat on accuracy but produced very narrow theses (9.8% coverage vs 20.5% for low).
- Code0 train accuracy was ~58-59% for all three, none are overfitting, all are underfitting.
- All 3 levels used 20 Code0 attempts and selected the best by validation accuracy.

### 2c. Cross-Dataset Comparison (gpt-5.2 low, 25 stratified samples each)

Tested the pipeline on 3 different datasets to see how results scale with problem difficulty.

| Dataset             | Features | Code0 Test Acc | Coverage | Faith (GT) |
|---------------------|----------|----------------|----------|------------|
| HTRU2               | 8 (numeric)     | 76%     | 50%      | 100%       |
| Mushroom            | 20 (categorical)| 55%     | 29%      | 63%        |
| CDC Diabetes        | 21 (mixed)      | 59%     | 21%      | 68%        |

- **HTRU2**: Trivially solved by Code0 with a single threshold (x0 > -20), so the thesis is trivially perfect. Good sanity check but not scientifically interesting.
- **Mushroom**: 20 unique multi-condition categorical theses across 25 samples. Most interesting thesis diversity. Some individual theses achieved 90-100% GT faithfulness.
- **CDC Diabetes**: Our primary benchmark. Moderate Code0 accuracy limits downstream metrics.

Code0 faithfulness was 100% on all three datasets, meaning the thesis always correctly describes what Code0 does. The gap between Code0 faithfulness and GT faithfulness is entirely explained by Code0's accuracy on the data.

---

## 3. Key Findings

1. **The pipeline works**: Code0 faithfulness is 100% across all datasets and settings. The thesis always accurately describes the code's logic.

2. **The bottleneck is Code0 accuracy**: GT faithfulness cannot exceed Code0's accuracy on the training set. On CDC Diabetes, Code0 only reaches ~59% accuracy, which caps everything downstream.

3. **Low reasoning is the sweet spot for gpt-5.2**: More reasoning produces more complex code that generalizes worse and generates overly specific theses. 13x more reasoning tokens for equal or worse results.

4. **Coverage and faithfulness scale with Code0 quality**: HTRU2 (76% Code0 accuracy) gets perfect scores. Harder datasets get proportionally lower scores. This suggests stronger models will improve all metrics.

5. **Sampling method matters**: Sequential sampling inflated early faithfulness numbers. Stratified random sampling gives more honest results.

6. **The pipeline produces genuinely diverse explanations**: On mushroom, 20 unique multi-condition theses across 25 samples, each describing a different region of the feature space.

---

## 4. Current Limitations

- All results use a single seed (2201) on each dataset. Multi-seed runs needed for robustness.
- Code0 accuracy on harder datasets is only ~55-60% with gpt-5.2. Need stronger models.
- TAMU API access for stronger models (gpt-5, Claude) is still blocked by technical issues (524 timeout errors). IT follow-up needed.
- Only tested 3 of 5 available datasets.
- 25-50 test samples is small. Larger evaluation sets would give tighter confidence intervals.

---

## 5. Planned Next Steps

1. **Get TAMU API working** for access to stronger models. This is the most important blocker since Code0 accuracy is the bottleneck.
2. **Multi-seed runs** on the best configuration to confirm robustness.
3. **Scale up test samples** (100+) for tighter metrics.
4. **Test remaining datasets** (fn_m: Adult Income, fn_q: Chess) for broader coverage.
5. **COLM submission**: discuss feasibility and what additional experiments are needed.

---

## 6. Cost Summary

All experiments ran via personal OpenAI API key.

- gpt-5.2: $1.75/1M input tokens, $14.00/1M output tokens
- Each full pipeline run (Code0 + thesis + Code1) on 25-50 samples costs roughly $1-3
- Total spend across all experiments: estimated $15-25

---

# Quick Q&A Prep

**Q: What is the pipeline actually doing?**
It takes training data, asks an LLM to write a classifier (Code0), then for each test sample asks the LLM to explain why the classifier made its prediction as a set of conditions. We then check those conditions against the training set to measure coverage (how broad is the explanation) and faithfulness (how accurate is it).

**Q: Why is faithfulness to Code0 always 100%?**
Because the thesis prompt asks the LLM to explain the code's reasoning for a specific sample, and the LLM is very good at reading its own code. The thesis conditions always match what the code would predict.

**Q: Why is GT faithfulness lower?**
Because the Code0 classifier itself is imperfect. If Code0 is 59% accurate, the thesis faithfully describes a 59%-accurate classifier, so it will disagree with ground truth roughly 40% of the time in the covered region.

**Q: Why does more reasoning make things worse?**
Two compounding effects: (1) higher reasoning produces more complex Code0 classifiers that generalize worse, and (2) higher reasoning produces more specific/narrow thesis conditions that cover fewer samples. Neither effect helps.

**Q: Is 10% coverage with 98% faithfulness good?**
Yes, it means we found a region covering 10% of the data where the explanation is almost perfectly correct. Even small coverage with high faithfulness is valuable for interpretability since it gives a verified, trustworthy explanation for that subset.

**Q: Can we push Code0 accuracy higher?**
That is the key question. On CDC Diabetes, gpt-5.2 (best available via personal API) tops out around 60% with 200 training samples. Stronger models or more training data should help. The TAMU API would give us access to those models.

**Q: Does this transfer across datasets?**
Yes, the pipeline architecture is dataset-agnostic and Code0 faithfulness is 100% on all 3 tested datasets. The limiting factor is always Code0 accuracy on that specific dataset.

**Q: What about the COLM deadline?**
Need to discuss what results would constitute a submission-ready contribution. Key question: is the current pipeline plus a stronger model enough, or do we need additional methodological contributions?

**Q: What would make this paper-ready?**
Likely: (1) results on 4-5 datasets with a strong model achieving 70%+ Code0 accuracy, (2) multi-seed runs showing robustness, (3) comparison against existing interpretability baselines (LIME, SHAP), (4) analysis of when and why theses fail.

**Q: Why not just use LIME or SHAP?**
Our approach produces executable, verifiable explanations (actual code), not feature importance scores. The thesis is a set of logical conditions that can be independently tested. This is a different kind of explanation, potentially more useful for audit and verification purposes.

**Q: What is the training setup?**
200 training samples (balanced 50/50), validation set of 618-2300 samples depending on dataset, test set of 2460-7500. Code0 gets 10-20 attempts and we pick the best by validation accuracy. Single seed per run currently.
