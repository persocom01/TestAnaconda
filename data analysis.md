1. Check out the .head() and .tail() of the data to get a sense of what they look like.
2. Check for nulls and decide how to handle them (Drop? Fill? With what?).
3. Check the data types of the columns using .dtypes() and adjust as necessary.
4. Remove any columns you're not interested in including and make sure the names of remaining columns are clear.
5. Get the summary statistics of the data using .describe().

Standard error
The standard error = standard deviation / sqrt (sample size)
It describes the dispersion of the sample statistic (such as mean)
around the population mean, giving you an idea whether more samples
are needed or not.
SSR = sum(prediction - mean)^2
SST = sum(y - mean)^2
SST = sum(y - prediction)^2

Project objective.
1. Business context.
Industry and scope of project.
2. Business objective.
Comprises the output of the project (best something), for a business goal (revenue), on a target demographic (customer).
3. Data objective.
Where to get the data?
Give a solid definition as to what you are looking for.
Fulfil objectives within a reasonable timeframe.
4. Evaluation.
How to know it was successful?

Overview:
1. Project objective.
2. Business overview and process.
3. Tech overview.
4. Key findings.
Highlight steps which lead to final conclusion.
Highlight unexpected results.
5. Key problems.
6. Conclusion.

For a confidence interval, our z score multiplier is 1.96. The number 1.96 comes from a standard Normal distribution.
The area under the standard Normal distribution between -1.96 and +1.96 is 95%.
For 90% confidence, use 1.645.
For 99% confidence, use 2.576.

p value is the probability of a data distribution occurring naturally given the null hypothesis is true.

A model with inputs over time t is considered usable over the next t/3 or t/5 periods. It gets more and more inaccurate as time passes, in what is called model decay.

model choice:
1. speed - the time needed by the model to process data and make predictions.
2. explainability - whether the model can be easily explained.
3. performance - how accurate the model is.

the 5 purposes of data quality processing are:
1. Validity
Are the values out of range?
Are null values acceptable?
Are the datatypes correct?
2. Accuracy.
Does the data conform to an objective true value?
3. Completeness.
Is data missing?
4. Consistency.
Is there duplicate data?
Concurrency (multiple user access) issues.
5. Uniformity.
Same timezone?
Same unit of measurement?
