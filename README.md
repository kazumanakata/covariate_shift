# covariate_shift

This is a repository for implementing introductory importance weghting [1, 2] for covariate shift adaptation in python.

In statistical machine learning, it is assumed that training and test samples are drawn from the same distribution.
However, in real world applications, this assumption may be violated due to the passage of time or any other reasons.
To make machine learning sysmtem robust to such changes, we can put weights on each trainging sample, indicating their importance for enabling predictors to generalize during training.

The importance of each training sample is derived from the ratio of test sample distribution to training sample distribution.
Training samples with high ratio values are likely to be more informative, possessing features also present in test samples.
By assigning an importance weight to each training sample, predictors can identify and prioritize the samples most relevant for test data during training.


このリポジトリでは、共変量シフトにおける基本的な重要度重みづけをpythonで実装します。

統計的機械学習では、訓練サンプルとテストサンプルが同じ分布から抽出されることが前提とされています。
しかし、実世界に機械学習を適用する際、時間の経過やその他の理由により、この前提が満たされないことがあります。
このような変化に対して機械学習システムを頑健にするために、各訓練サンプルに重みを付けることができます。
この重みは、各訓練サンプルがどの程度重要かを示し、予測器の汎化性能を向上させるために使用することができます。

各訓練サンプルの重要度は、テストサンプルの分布と訓練サンプルの分布の密度比から導出されます。
高い密度比を持つ訓練サンプルは、テストサンプルにも存在する特徴を持っている可能性が高く、より学習に有効であると考えられます。
各訓練サンプルに重要度（密度比）を割り当てることで、予測器はテストデータに最も関連するサンプルを学習時に識別し、優先して学習に使用ことができます。


<img src="https://github.com/kazumanakata/covariate_shift/assets/121463877/bc1690e2-411b-4727-8c6a-91750641a215"><br>
A target function and a distribution function.<br>
真の関数とサンプル分布関数。

<img src="https://github.com/kazumanakata/covariate_shift/assets/121463877/0c8f0bda-51de-4da2-a31c-b2dc29152f40"><br>
Train and test sample distributions.<br>
訓練とテストサンプルの分布。

<img src="https://github.com/kazumanakata/covariate_shift/assets/121463877/d18ce528-d130-4914-b95e-21dd5f3f2325"><br>
Plots of train and test samples.<br>
訓練とテストサンプルのプロット図。

<img src="https://github.com/kazumanakata/covariate_shift/assets/121463877/f68258b9-0a74-4e5e-8b1e-764a50f663e7"><br>
The importance weight derived from the train and test distribution.<br>
訓練とテストサンプルの分布から得られる重要度(密度比)。

<img src="https://github.com/kazumanakata/covariate_shift/assets/121463877/7e4410fe-d1a6-4e35-a14d-774b759e9d36"><br>
Linear functions after SGD(Stochstic Gradient Descent) w/ and w/o importance weight adaptation.<br>
確率的勾配降下法による重要度重みづけを用いた時と用いないときの線形近似の結果。

## Reference
1. Hidetoshi Shimodaira: Improving predictive inference under covariate shift by weighting the log-likelihood function., Journal of Statistical Planning and Inference Volume 90, Issue 2, 1 October 2000, Pages 227-244.
1. 杉山 将, 山田 誠, ドゥ・プレシ マーティヌス・クリストフェル, リウ ソン: 非定常環境下での学習：共変量シフト適応，クラスバランス変化適応，変化検知., 日本統計学会誌, vol.44, no.1, pp.113{136, 2014.
