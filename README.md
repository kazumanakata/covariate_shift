# covariate_shift

This is a repository for implementing introductory importance weghting for covariate shift adaptation in python.

In statistical machine learning, training and test samples are required to be drawn from the same distribution.
However, this assumption can be degraded in real world applications due to the passage of time or any other reasons.
In order to make machine learning sysmtem robust to such changes, we can put weights on each trainging sample, indicating how each training sample is important for predictors to generalize.
