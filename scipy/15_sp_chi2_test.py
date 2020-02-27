# Demonstrates the use of the scipy chi2 test in hypothesis testing.
# The chi2 test is used on a categorical x and a categorical y to determine if
# there is enough evidence to reject the null hypothesis that there is no
# relationship between them.
# On testing, it appears even very weak correlations will cause the null
# hypothesis to be rejected. Use with a pinch of salt.
import pandas as pd
import scipy.stats as stats

df = pd.read_csv('./datasets/mushrooms.csv')
print(df.head())
x = df['cap-shape']
y = df['class']
con_table = pd.crosstab(x, y)
# A contingency table displays the observed frequency of values.
observed_values = con_table.values
print('observed_values:')
print(observed_values)
print()

chi2, p, dof, expected_values = stats.chi2_contingency(con_table)
print('expected_values:')
print(expected_values)
print()

no_of_rows = len(con_table.iloc[0:, 0])
no_of_columns = len(con_table.iloc[0, 0:])
degrees_of_freedom = (no_of_rows-1)*(no_of_columns-1)
print('degrees_of_freedom:')
print('calcuated -', degrees_of_freedom)
print('function -', dof)
print()

# Calculation of chi2.
chi_square = sum(
    [(o-e)**2./e for o, e in zip(observed_values, expected_values)])
chi_square_statistic = sum(chi_square)
print('chi_square_statistic:')
print('calcuated -', chi_square_statistic)
print('function -', chi2)
print()

# Critical chi2 value for null hypothesis to be rejected.
alpha = 0.05
critical_value = stats.chi2.ppf(q=1-alpha, df=degrees_of_freedom)
print('critical_value:', critical_value)
print()

# p-value
pval = 1 - stats.chi2.cdf(x=chi_square_statistic, df=degrees_of_freedom)
print('pval:')
print('calcuated -', pval)
print('function -', p)
print()

null = 'there is no relationship between the two categorical variables'
if chi_square_statistic >= critical_value:
    print(
        f'chi2 = {chi_square_statistic} >= {critical_value}, therefore reject null hypothesis that {null}.')
else:
    print(f'chi2 = {chi_square_statistic} < {critical_value}, therefore unable to reject null hypothesis that {null}.')
print()

if pval <= alpha:
    print(
        f'pval = {pval}, therefore reject null hypothesis that {null}.')
else:
    print(
        f'pval = {pval}, therefore unable to reject null hypothesis that {null}.')
print()
