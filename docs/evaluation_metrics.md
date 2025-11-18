# Evaluation Metrics

Evaluation metrics are required to evaluate the model for various reeasons such as if the chossen layers, model are apt for the problem being solved, if it can be improved, if the model is being overfitted, etc.

## Confusion Matrix

- Table with the correct number of correct and incorrect predictions broken by class.
- Makes it easy to see if the classifier is confusing the classes.
- It isn't a performance measure by itself but the values inside it tell a story.
- The values in the matrix are dependent on the amount of data in each class which is why it is better to normalise the data.

## Accuracy

- Ratio of correct predictions to total samples.
- It isn't a good metric for imbalanced datasets.

$$
accuracy = \frac{Correct\space Predictions}{Total \space Samples}
$$

## Precision

- Fraction of relevant instaces among the instances classified as relevant.
- Helps in decreasing the rate of False Positives.

$$
precision = \frac{True Positive}{True Positive+FalsePositive}
$$

## Recall/Sensitivity

- Fraction of the total amount of true positives that were retrieved.
- Should be used with Precision for best results.
- Also known as True Positive Rate.

$$
recall = \frac{True Positive}{True Positive+FalseNegative}
$$

## F1 Score

- Combination of Precision and Recall.
- Best value of 1, which is reaches when recall and precision are 100%.
- Used when working on a small positive class.

$$
F_1 = 2\frac{Precision * Recall}{Precision + Recall}
$$

## ROC Curve

- Receiver Operating Characteristic(ROC) Curve.
- Graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
- Helps select the right threshold.
- The best ROC curve is one that is as close to the upper-left corner as possible.
