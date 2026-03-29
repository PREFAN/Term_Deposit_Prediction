# Term_Deposit_Prediction
Project Overview

This project focuses on applying statistical analysis and machine learning to predict whether a customer will subscribe to a term deposit offered by a bank. Financial institutions use telemarketing campaigns to promote products such as term deposits, but campaign success depends heavily on identifying the right customers. The goal of this project is to help banks improve marketing efficiency by predicting likely subscribers using customer and campaign-related features.

Using the Portuguese Bank Marketing Dataset from the UCI Machine Learning Repository, the project performs a comprehensive analysis of customer behavior and telemarketing campaign outcomes. It combines exploratory data analysis (EDA), probability and distribution analysis, hypothesis testing, Bayesian analysis, A/B testing, and logistic regression modeling to understand the data and build a predictive model.

The results show that the dataset is imbalanced, and although the variables do not show strong linear relationships, the logistic regression model achieved strong predictive performance. The model produced an accuracy of 82.1% and demonstrated excellent performance based on the ROC-AUC curve, making it a useful tool for helping banks identify potential customers for term deposit subscription and improve telemarketing campaign strategy.

Problem Statement

Banks rely on telemarketing as a major marketing strategy to promote financial products such as term deposits. However, contacting every customer is costly, time-consuming, and often inefficient. A better approach is to identify which customers are most likely to subscribe before launching a campaign.

The challenge is that customer subscription behavior depends on multiple factors, including:

Customer demographics
Financial status
Previous marketing contact history
Call duration
Communication channel
Campaign characteristics

This project addresses the challenge by building a predictive logistic regression model that estimates the likelihood of a customer subscribing to a term deposit based on relevant features from historical campaign data.

Project Objectives

The major objectives of this project are to:

Perform exploratory data analysis on the bank marketing dataset
Understand the statistical distribution of customer and campaign variables
Analyze relationships between customer features and subscription outcomes
Conduct probability and distribution analysis
Perform inference and hypothesis testing
Apply Bayesian analysis to estimate subscription probability
Conduct A/B testing to compare marketing strategies
Build and evaluate a logistic regression model
Predict whether a customer will subscribe to a term deposit
Provide recommendations to improve bank telemarketing efficiency and customer targeting
Dataset Information

This project uses the Bank Marketing Dataset from the UCI Machine Learning Repository.

Dataset Details
Source: UCI Repository (bank-full.csv)
Country/Context: Portuguese Bank
Rows: 45,211 customer campaign records
Columns: 17 features
Target Variable: y (whether the client subscribed to a term deposit)
Target Variable
yes = customer subscribed
no = customer did not subscribe

For modeling purposes:

yes → 1
no → 0

The target column was renamed to success during preprocessing.

Feature Categories

The variables in the dataset can be grouped into four major categories:

1. Customer Demographics
Age
Job
Marital status
Education
Default
2. Client Financial Information
Balance
Housing loan
Personal loan
3. Campaign / Contact Information
Contact type
Day
Month
Duration
Campaign
Pdays
Previous
Poutcome
4. Target Variable
y / success (term deposit subscription outcome)

These features help the model learn which types of customers are more likely to subscribe.

Tools and Technologies Used

This project was implemented entirely in Python.

Programming Language
Python
Libraries / Tools
Pandas – data loading and preprocessing
NumPy – numerical operations
Matplotlib – data visualization
Seaborn – statistical plots and heatmaps
SciPy – hypothesis testing and statistical functions
Statsmodels / Scikit-learn – logistic regression and model evaluation
Jupyter Notebook / Python scripts – analysis environment
Methodology

The project was divided into six major analytical sections before fitting the final predictive model.

1. Exploratory Data Analysis (EDA)

EDA was used to understand the structure, distribution, and patterns in the dataset.

Analyses included:

Descriptive statistics
Measure of central tendency and spread
Correlation matrix heatmap
Scatter plots
Histogram plots
Pie charts for categorical variables
2. Probability and Distribution Analysis

This section examined:

Joint probability
Marginal probability
Conditional probability
Multinomial probability modeling

The variables Marital and Education were used to understand distribution and dependency patterns.

3. Inference and Hypothesis Testing

Statistical tests performed include:

Two-sample t-test
Chi-square goodness-of-fit test
95% Confidence Interval estimation

These tests were used to determine whether differences in customer groups and campaign outcomes were statistically significant.

4. Bayesian Analysis

Bayesian inference was used to estimate:

Posterior distribution of customer subscription
Probability that a customer subscribes to a term deposit
95% credible interval for subscription rate
5. A/B Testing

A/B testing was used to compare the effectiveness of:

Phone calls
Email marketing

This helped assess which telemarketing strategy produced better campaign outcomes.

6. Logistic Regression Modeling

Finally, a logistic regression model was built to predict customer subscription.

Steps included:

Encoding categorical variables using one-hot encoding
Splitting data into:
80% training
20% testing
Fitting the logistic regression model
Evaluating performance using:
Confusion matrix
Accuracy
Precision
Recall
F1-score
ROC curve
AUC score

All analyses and modeling were implemented in Python.

Exploratory Data Analysis (EDA) Highlights

Key findings from the EDA section include:

The dataset contains no missing values
It includes both numerical and categorical variables
The distribution of variables is non-uniform
Many numeric variables are right-skewed
Scatter plots showed little to no strong linear relationship
The correlation matrix showed mostly weak correlations
Two notable relationships were:
duration and success (moderate positive relationship)
pdays and previous (moderate positive relationship)
Notable Statistical Insights
Minimum age: 18
Maximum age: 95
Mean age: 40.93
Mean balance: 1362.27
Subscription success rate: approximately 11.7%

These findings suggest that the dataset is broad, skewed, and imbalanced, which directly affects model behavior.

Probability and Distribution Findings

Using Marital and Education:

Married customers had the highest marginal probability
Secondary education had the highest frequency among education groups
Joint probability showed married + secondary education as the most common combination
Conditional probability suggested:
Single customers tend to have relatively higher tertiary education
Marital status varies more as education level increases

A multinomial model was also fitted and was found to be highly effective for representing the distribution between marital status and education categories.

Inference and Hypothesis Testing Results
Two-Sample t-test

A two-sample t-test comparing campaign success between management and technician job groups found:

p-value < 0.05
This indicates a statistically significant difference in campaign success between the two job types
Chi-Square Goodness-of-Fit

A chi-square test on marital status found:

p-value < 0.05
This indicates the marital distribution is statistically significant
95% Confidence Interval

The estimated 95% confidence interval for the campaign success rate was:

0.11 to 0.12

This means the true term deposit subscription rate in the population is likely between 11% and 12%.

Bayesian Analysis Results

Bayesian inference estimated the probability that a randomly selected customer subscribes to a term deposit.

Results
Estimated subscription probability: 0.117 (11.7%)
Posterior distribution: Beta(5290, 39923)
95% credible interval: 0.11 to 0.12

This aligns closely with the classical confidence interval and confirms the relatively low overall subscription rate in the dataset.

A/B Testing Results

The project compared phone calls and email marketing as telemarketing strategies.

Key Findings
Phone calls had a slightly higher success rate than emails
However:
p-value > 0.05
The difference was not statistically significant
The observed difference in success rate was very small (0.0012)

Although the statistical test did not prove a significant advantage, the project concluded that phone calls may still be more practical and effective due to faster customer interaction and decision-making.

Machine Learning Model
Logistic Regression for Term Deposit Prediction

The final predictive model used in this project is:

Logistic Regression
Why Logistic Regression?

Logistic regression is well suited for:

Binary classification
Interpretable coefficients
Predicting probability of customer subscription
Business applications where probability estimates are useful for decision-making
Preprocessing for Modeling
Target variable converted from categorical to numeric
Categorical variables encoded using one-hot encoding
Train-test split:
80% training
20% testing
Model Performance

The logistic regression model performed well overall, especially considering the imbalanced nature of the dataset.

Confusion Matrix
True Negative (TN): 6632
False Positive (FP): 1372
False Negative (FN): 243
True Positive (TP): 796
Class Distribution in Test Output
Class 0 (Failure): 8004
Class 1 (Success): 1039

This confirms the dataset is imbalanced, with many more non-subscribers than subscribers.

Evaluation Metrics
Overall Accuracy
82.1%
Class-wise Metrics
Class 0 (Failure / No Subscription)
Precision: 96%
Recall: 83%
F1-score: 89%
Class 1 (Success / Subscription)
Precision: 37%
Recall: 77%
F1-score: 50%
Interpretation
The model performs strongly for the majority class (non-subscribers)
It captures many actual subscribers (good recall for class 1)
But precision for subscribers is lower due to imbalance
This means the model may produce more false positives for likely subscribers

Even with this limitation, the model is still useful because it helps identify potential target customers for telemarketing.

ROC-AUC Performance

To go beyond accuracy, the project also evaluated the model using:

ROC Curve
AUC (Area Under the Curve)
Result

The project concluded that the ROC-AUC performance was excellent, indicating that the logistic regression model can effectively separate likely subscribers from non-subscribers despite the class imbalance.

Key Findings

This project produced several important findings:

The dataset is imbalanced
There are no missing values
Many variables are non-uniformly distributed and right-skewed
Most variable pairs show weak or no linear correlation
duration shows a moderate positive relationship with subscription success
The overall term deposit subscription rate is low (~11.7%)
Bayesian analysis confirms the low but measurable probability of subscription
A/B testing found no statistically significant difference between phone and email strategies
Logistic regression achieved 82.1% accuracy
The model performs strongly overall and is particularly useful for customer targeting in telemarketing campaigns
Business Relevance

This project is highly relevant to the banking and financial services industry.

Who benefits from this project?
Banks
Telemarketing teams
Sales operations teams
Customer relationship managers
Marketing analysts
Business intelligence teams
Data analysts / data scientists in finance
Why it matters

The model can help banks:

Identify customers more likely to subscribe
Reduce wasted telemarketing effort
Save time and resources
Improve campaign efficiency
Increase conversion rates
Make more data-driven marketing decisions

This is especially useful when dealing with large customer databases where contacting every customer is inefficient.

Project Limitations

The project also identifies important limitations:

The dataset is imbalanced
The model is biased toward the majority class
The model performs better for non-subscribers than subscribers
There are many unknown values in the dataset
The dataset is from a Portuguese bank, so results may not generalize globally

These are important considerations for real-world deployment.

Recommendations / Future Improvements

The project recommends several ways to improve model performance:

Adjust the default probability threshold (0.5) to optimize the F1-score
Apply SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance
Use undersampling for the majority class
Explore more advanced models such as:
Random Forest
XGBoost
Compare linear and non-linear classifiers
Improve handling of unknown values and feature engineering

These improvements could increase predictive performance, especially for the minority class (actual subscribers)
