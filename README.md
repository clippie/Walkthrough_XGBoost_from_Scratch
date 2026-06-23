# XGBoost Math (from scratch)

Decision trees are algorithms that can be used to make predictions based on a series of decisions. For example, if you wanted to predict if someone was more likely to eat dinner at home or a restaurant, you could have a decision tree like this:
![Decision Tree](./screenshots/DecisionTree.png)
Based on this decision tree, if it is Tuesday and they have more than 10 ingredients at home, you would predict that the person would eat at home. A collection of decision trees with random sampling and feature selection can be called a random forest.

A boosted tree is different from random forests because it builds "smarter" trees that learn sequentially, each one correcting the errors of the previous.

XGBoost is a gradient boosted decision tree algorithm that offers several advantages over basic decision trees, random forests, and basic gradient boosted trees. These advantages have made XGBoost one of the most popular machine learning algorithms over the past 10 years. In this example, I will go over some of the basic math of XGBoost and outline what makes this algorithm special.

## Full Notebook
![Full Notebook](./screenshots/Full.png)

## Setup
For this example, we will need some labeled data and to set some parameters. To keep the calculations manageable, there will only be 6 rows and 3 features. Every unique value will require its own calculation down the line, so it is better to limit them now. Here, the data shows if a person was accepted for a position with features including age, the presence of a cover letter, and the number of skills. The dataset is small, but I made sure to add in some patterns to replicate what might be seen in real data.

The parameters that will be used throughout are lambda (L2 regularization), gamma (leaf penalty), and eta (learning rate/shrinkage size). I also set the max depth at 3, but that will not come into effect here. L2 regularization is one of the things that makes XGBoost different from other boosted trees. Regularization is used to prevent models from overfitting on the specifics of training data and instead force it to generalize. XGBoost's core objective uses L2 regularization. It also supports optional L1 regularization via the alpha parameter, but this defaults to 0 and is not part of the foundational algorithm. L1 regularization penalizes the absolute value of each weight, which has a tendency to push weights to 0. This can be helpful with high-dimensional data, but since this example only has 3 features, L1 regularization will not be used. L2 regularization penalizes the square of each weight, meaning larger weights get punished disproportionately hard, pushing the model toward smaller, more distributed corrections. A leaf penalty is used to set the ["minimum loss reduction required to make a further partition on a leaf node of the tree"](https://xgboost.readthedocs.io/en/stable/parameter.html), meaning that a larger leaf penalty should result in fewer splits because the tree needs to be more sure that each split is leading to a positive gain in accuracy. Learning rate/shrinkage size is used to adjust the effect of feature weights, with the goal of spreading out updates over more iterations.

The default values for these parameters are lambda = 1, gamma = 0, and eta = 0.3. I have chosen to use these default values, with the exception of eta, which I have increased to 0.5 to show a larger change over only one tree.

![Start](./screenshots/Start.png)

## Step 0: Initial Prediction
Unlike a basic decision tree, the first step is to make the initial prediction. We are predicting a binary variable, so we can use log odds. Log odds is helpful for a binary classification problem like this because XGBoost needs continuous, unbounded values to iterate on. Each iteration adds a correction that could be any positive or negative, large or small number. Probabilities are bounded between 0 and 1, so applying unbounded corrections directly to a probability would quickly produce nonsensical values like 1.3 or -0.2. Therefore, these updates are given to the log odds value, which is unbounded. The result can then be sent into a sigmoid function to convert it back into a probability, which can be used to calculate the next step.

**Log in this equation refers to natural log (ln)*

![Step 0](./screenshots/Step0.png)

## Step 1: Calculate Gradients and Hessians
XGBoost uses both the 1st and 2nd derivative of the loss function. The 1st derivative (gradient) is the slope of the loss function at a prediction. This tells you the direction and magnitude of error. The 2nd derivative (hessian) measures the curvature of the loss function at a prediction. A higher curvature means the loss is changing rapidly, so corrections should be more conservative. These values can be calculated using the output from the sigmoid function in the previous step.

![Step 1](./screenshots/Step1.png)

## Step 2: Find the root split
Now we can actually start to build the tree. This is done by evaluating the possible splits and choosing the one with the largest gain, using the gain equation from [(Chen, Guestrin, 2016)](https://arxiv.org/pdf/1603.02754):

![Gain Equation](./screenshots/equation7.png)

**For most practical applications, XGBoost will not actually evaluate on every candidate split. Instead, it uses other techniques like binning and column subsampling. For this example, it makes sense to evaluate all candidate splits.*

The equation above can be simplified and used on all the candidate splits. GL is gradient left, HL is hessian left, and the same for the right. Lambda also makes an appearance here to aid in the regularization that was mentioned earlier. Let's take the GL^2/(HL+lambda) portion of the equation. If there are a lot of examples on the left side of a split, for example, 400 * 0.25, this would be HL = 100. Adding a lambda of 1 does not have a relatively strong effect when it's used here. On the other hand, if there are not a lot of examples for a split, for example, 1 * 0.25, which would be HL = 0.25, then adding a lambda of 1 will drastically change the divisor. This is one of the strengths of L2 regularization, in that it affects smaller, more likely to overfit splits more heavily than larger ones.

![Step 2](./screenshots/Step2.0.png)

Once all the candidate splits are calculated, choosing the best split is as simple as finding the highest positive gain. In this case, it is Skills less than or equal to 7.

![Step 2](./screenshots/Step2.5.png)

## Step 3: Depth 2
The root split is the best separator of the entire dataset. The next split is the best separator of the already partitioned data. Given the split of skills less than or equal to 7, people 3 and 6 are on the left, and people 1, 2, 4, and 5 are on the right. Since none of the candidate splits on the left account for a positive gain, there are no logical further splits for this branch, meaning this node turns into a leaf and ends the branch. The right side's largest gain for a candidate split is age less than or equal to 28, with a gain of 0.4928571429. This is a good example of how the leaf penalty (gamma) value works. Here, gamma is 0, but if that value were set to 0.5, for example, the gain would turn negative, meaning that there would be no positive gain for this branch, and it would become a leaf. Since this is not the case, age less than or equal to 28 becomes the next split on the right, as shown in the figure.

![Step 3](./screenshots/Step3.png)

## Step 4: Depth 3
Looking at the data, we can see that it's already fully split, but I did the math anyway. As expected, all the gains are negative, and the nodes become leaves. Since the max depth was set to 3 at the beginning, this layer was the last chance to find splits, and the negative gains end the tree anyway.

![Step 4](./screenshots/Step4.png)

## Step 5: Calculate Leaf Weights
Using the equation from [(Chen, Guestrin, 2016)](https://arxiv.org/pdf/1603.02754), we can use the gradients and hessians from each split to calculate the weights:

![Equation 5](./screenshots/equation5.png)

Again, you can see the use of lambda in this equation and how it disproportionately affects lower-hessian (fewer data points or more confident predictions). The weights generated from this equation will then be used in the next step to make updated predictions.

![Step 5](./screenshots/Step5.png)

## Step 6: Update prediction

Now that the tree is fully grown, we can send each example through it like a normal decision tree. However, instead of getting a prediction at the terminal node, we get a weight, which is used to update the original prediction. To get the new raw score, the original log odds value is used as opposed to the post-sigmoid probability. This is because of the bounds requirement that was mentioned earlier. You can also see the effect of the learning rate/shrinkage size here, and how it dilutes the weights' impact on each update. While the new raw score is unbounded, it is not as interpretable as a probability, so we send it back through the sigmoid function to get the updated predictions. These updated predictions are shown in the table on the right. All the predictions moved in the right direction, with the biggest change being the skills > 7, age <= 28 group. This is because all three people that got accepted ended up in a leaf together on the left side of the age less than or equal to 28 split, which produced the largest leaf weight. If the learning rate/shrinkage size were increased, then there would be an even larger change. To start the next tree, these new prediction values would be used to calculate the updated gradients and hessians, and the candidate splits would be evaluated again.

![Step 6](./screenshots/Step6.png)


Despite having some similarities to the simple decision tree shown at the beginning, XGBoost is actually quite different, as shown in this example. These differences extend beyond this example. The real XGBoost implementation adds engineering optimizations that make it viable at scale, yet the core math remains the same. This project completely changed my perspective on how XGBoost works, and it is easy to understand how this algorithm got so popular with all the advantages like regularization, leaf penalties, and 2nd order derivatives. Together, these properties make XGBoost particularly effective on structured tabular data, where it consistently outperforms simpler tree methods and often competes with deep learning.

---

© Caden Lippie 2026. Licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
