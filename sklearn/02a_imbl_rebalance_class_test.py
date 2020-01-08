# The first thing we need to ask before adjusting the relative class
# frequencies is should be balance the classes. Balancing classes loses
# information about relative class frequencies, which makes the model
# generalize between classes. In effect, the model will overestimate the
# minority class and underestimate the majority. This may or may not be a good
# depending on the problem at hand. Some cases when balancing classes is
# desirable are:
# 1. When the cost of failing to predict the minority class is high, such as
# predicting cancer cases or fraud.
# 2. When the number of minority class is too small for the model to trained
# properly, and balancing the classes might improves model metrics like
# precision/recall.
# 3. When it is known that the distribution of classes in the sample is
# different from that of the population, thus balancing classes actually
# increases the accuracy of the model on the population.
from imblearn.over_sampling import SMOTE
