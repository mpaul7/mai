## Resource 



```
https://www.geeksforgeeks.org/applying-l2-regularization-to-all-weights-in-tensorflow/
https://towardsdatascience.com/impact-of-regularization-on-deep-neural-networks-1306c839d923
https://www.v7labs.com/blog/overfitting
```
Ref: https://chatgpt.com/c/6793cb7b-5fcc-8013-8ab4-293ba5173a0e
L2 regularization is different from L1 regularization (also known as Lasso regularization). While L2 regularization shrinks the coefficients towards zero, L1 regularization can set some coefficients exactly to zero, effectively performing feature selection.

Elastic Net Regularization

Elastic Net regularization is a combination of L1 and L2 regularization. It incorporates both the L1 penalty (Lasso) and the L2 penalty (Ridge). This can be useful when there are multiple correlated features, as it can select one feature from a group of correlated features while shrinking the coefficients of the others.

REgularization:
L2 regularization, also known as Ridge Regression, is a technique used in machine learning to prevent overfitting by adding a penalty term to the cost function. It helps to shrink the coefficients of the model towards zero, reducing the impact of less important features.

In L2 regularization, the penalty term added to the cost function is the sum of the squares of the coefficients multiplied by a regularization parameter (alpha). This term penalizes large coefficients, encouraging the model to learn smaller, more generalizable coefficients.
The regularization strength, alpha, controls the amount of regularization applied to the model. A larger alpha value will result in stronger regularization, shrinking the coefficients more aggressively towards zero. Choosing the right alpha value is crucial for finding the balance between underfitting and overfitting.

L2 regularization is commonly used in linear regression to prevent overfitting when dealing with high-dimensional data or when there is multicollinearity among the features.

L2 regularization can also be applied to logistic regression, which is a classification algorithm. It helps to prevent overfitting by penalizing the magnitude of the coefficients, similar to linear regression.

L2 regularization can be applied to neural networks to reduce overfitting by adding a penalty term to the cost function based on the squared weights of the network. This encourages the weights to be smaller and helps to prevent the network from memorizing the training data.

L2 regularization is different from L1 regularization (also known as Lasso regularization). While L2 regularization shrinks the coefficients towards zero, L1 regularization can set some coefficients exactly to zero, effectively performing feature selection.

Elastic Net regularization is a combination of L1 and L2 regularization. It incorporates both the L1 penalty (Lasso) and the L2 penalty (Ridge). This can be useful when there are multiple correlated features, as it can select one feature from a group of correlated features while shrinking the coefficients of the others.

In neural networks, early stopping can be used in conjunction with L2 regularization to prevent overfitting. Early stopping involves monitoring the validation loss during training and stopping the training process when the validation loss starts to increase, indicating that the model is beginning to overfit.

The regularization path is a useful visualization technique that shows how the coefficients of a regularized model change as the regularization strength (alpha) is varied. It can help in understanding the effect of regularization and selecting an appropriate alpha value.

Cross-validation is a technique used to evaluate the performance of a model and select the optimal hyperparameters, such as the regularization strength (alpha) in L2 regularization. It involves splitting the data into multiple folds and training and evaluating the model on different combinations of these folds.

L2 regularization helps to balance the bias-variance tradeoff in machine learning models. By adding the regularization term, it increases the bias of the model (underfitting), but reduces the variance (overfitting), leading to better generalization performance on unseen data.