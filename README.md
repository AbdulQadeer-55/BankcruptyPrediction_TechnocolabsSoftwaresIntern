<a name="br1"></a> 

# **Project Introduction**

In this project, we will make a Web App to be used in measuring and

predicting the financial conditions of a corporate entity, using various

statistical forecasting techniques like Logistic Regression and Naive

Bayes to predict whether they are going to bankruptced based on the some

features given.

The Polish bankruptcy dataset is a collection of financial data from

Polish companies that were active between 2000 and 2013. The dataset

includes information on 7,027 companies, of which 271 went bankrupt.

We performed: Data preprocessing, EDA, Feature Engineering, Modeling

and Deployment.

# **Dataset Overview**

At first, we were given five files of data, colleced from five different

years, The dataset is about bankruptcy prediction of Polish companies in

manufacturing sector, containing all their financial data and their final

status.

\- The label of the dataset is whether the stock price is \*\*Bankrupt\*\*

(labeled as \*\*1\*\*) or \*\*Not Bankrupt\*\*(labeled as \*\*0\*\*) on that day.

# **Preprocessing and Sentiment Analysis**

At first, we converted the arff to csv file. We firstly worked on the data

on separate files (5 files). Then, we checked for duplicates and dropped

them. Then, Change the data type of 'Classification' column to save up

memory to integer rather than object.

We replaced null values by performing median imputation on them.

Handling outliers from data using LOF by from sklearn.neighbors import

LocalOutlierFactor. We also used using isolated forest by from

sklearn.ensemble import IsolationForest.Finally, Saving the resulted

dataframe as a csv file.

# **EDA**

### **\*\*Introduction:\*\***

At first, we merged the separate five files into one csv file.

At first, we searched about bankruptcy and found out an equation called

'the Altman Z-Score'

Altman Z-Score = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E A

A = working capital / total assets

B = retained earnings / total assets

C = earnings before interest and tax / total assets

D = book value of equity / total liabilities

E = sales / total assets



<a name="br2"></a> 

A score below 1.8 means it's likely the company is headed for bankruptcy,

while companies with scores above 3 are not likely to go bankrupt.

We selected the features mentioned above and checked their correlation

with target variable using heatmap

### **\*\*Information of Dataset:\*\***

After the data was merged, it becomes **40768x65.**

At first, we selected the columns we are interested in based on a certain

threshold correlation value with the target column.

We used Heatmaps to visualize the correlation values of the selected

features with the target variable

Then we do the following visualizations on them:

### **\*\*Univariate Analysis:\*\***

Plotted

1-Scatter plots

2-Line plots

3-Histograms

4-Box plots

5-KDEto see the distribution of data for each column and found that few

variables are normally distributed. However, we can't really say about

that which variables needed to be studied. Since, Subjectivity and

polarity variable are derived ones and other historical stock variables

required to study more that how they are related to each other.

### **\*\*Bivariate Analysis:\*\***

Plotted

1-scatter plots vs target variable

2-pair plots on a random sample of the columns

### **\*\*Multi-variate Analysis:\*\***

1-Grouped Bar charts for a random sample of columns

2-3D scatter plots for a random sample of columns

3-Heatmaps

Also, we calculated the statistical summary of the features we are

interested in. We calculated their count,mean,min,quartiles,max,and

standard deviation.

Observed outliers in few categorical columns as well.

# **Preprocessing Again and Feature Engineering**

Now, after performing EDA, where the 5 datasets were concatenated, we

performed Feature Engineering process in order to remove the columns that

don’t have an impact on the target variable (Classification column).

First, we started doing correlation matrix with all the 64 columns against

the Target variable, but this was not clear. Hence, we decided to put a



<a name="br3"></a> 

threshold of absolute 0.8 and draw correlation matrix once again. 27

columns appeared with relation with the ‘Classification Column’.

We then performed another feature engineering technique which is PCA. We

followed these steps:

1\. STANDARIZATION: we used sklearn.preprocessing import StandardScaler

and scaled\_data is resulted.

2\. COVARIANCE MATRIX: It’s actually the sign of the covariance that

matters; if positive then: the two variables increase or decrease

together (correlated).If negative then: one increases when the other

decreases (Inversely correlated). We used numpy.

3\. COMPUTE THE EIGENVECTORS AND EIGENVALUES OF THE COVARIANCE MATRIX TO

IDENTIFY THE PRINCIPAL COMPONENTS: Principal components are new

variables that are constructed as linear combinations or mixtures of

the initial variables.These eigenvalues represent the amount of

variance captured by each principal component. You can use these

eigenvalues to calculate the explained variance ratio and decide how

many principal components to retain for dimensionality reduction. We

used linalg from numpy library.

4\. Feature Vector: Normalize eigenvalues to get explained variance

ratio. Then, we calculated the cumulative explained variance. We put

a threshold of 95% of the explained variance to exclude principle

components below that.

5\. RECAST THE DATA ALONG THE PRINCIPAL COMPONENTS AXES: the aim is to

use the feature vector formed using the eigenvectors of the

covariance matrix, to reorient the data from the original axes to

the ones represented by the principal components (hence the name

Principal Components Analysis).WE saved tha data in reduced\_data1.

The last technique we used was Mutual Information (MI):

Calculate Mutual Information for Each Feature:

Compute the Mutual Information score between each feature and the target

variable. Mutual Information measures the reduction in uncertainty of one

variable given the knowledge of another variable. In this case, it

measures how much knowing the value of a particular feature reduces the

uncertainty about the target variable.

We first imported pandas import pandas as pd then used sklearn package

from sklearn.feature\_selection import mutual\_info\_classif. MI scores were

resulted. Select Top Features: Choose a predefined number of top features

with the highest Mutual Information scores. Alternatively, you can set a

threshold value and select features that have Mutual Information scores

above that threshold. We chose to put a threshold of 0.005 and save the

resulted columns in a new dataframe called selected\_data1.


# **Model Building**

Metrics considered for Model Evaluation

We incorporated Testing time, Accuracy, Precision, F-1 score, support and

recall.

\- Accuracy: What proportion of actual positives and negatives is correctly

classified?

\- Precision: What proportion of predicted positives are truly positive ?

\- Recall: What proportion of actual positives is correctly classified ?

\- F1 Score : Harmonic mean of Precision and Recall

\- Testing time : in the context of model training, provides a measure of

how efficiently and accurately a trained model can make predictions on

new, unseen data

\- Support : refers to the number of instances or data points in a dataset

that belong to a specific class or category, providing an indication of

how well-represented that class is in the data.

## **Logistic Regression**

\- Logistic Regression is a statistical method used in machine learning to

model the relationship between a binary dependent variable (usually

representing two classes) and one or more independent variables. It helps

us understand how probabilities change based on various actions or

features.

\- The function is defined as P(y) = 1 / 1+e^-(A+Bx)

\- In logistic regression, the goal is to find the best-fit S-shaped curve

(S-curve) that models the relationship between the input features and the

probability of the binary outcome. This curve is characterized by the

values of A and B.

\- To optimize the logistic regression model, different solvers can be

used:

### LBFGS Solver (L-BFGS-B):
Limited-memory Broyden-Fletcher-Goldfarb-Shanno

(L-BFGS-B) is an optimization algorithm that works well for logistic

regression with a large number of samples. It's efficient for multiclass

problems and can handle L1 and L2 regularization.

### SAGA Solver: 
SAGA (Stochastic Average Gradient Descent) is an optimization

algorithm suitable for large datasets and is capable of handling both L1

and L2 regularization. It's particularly effective for solving large-scale

logistic regression problems.

### LIBLINEAR Solver: 
LIBLINEAR is a linear solver designed for linear

classification and regression tasks. It's efficient for large datasets and

is suitable for L1-regularized logistic regression.

### SAG Solver: 
SAG (Stochastic Average Gradient) is another optimization

algorithm suitable for logistic regression. It's known for its efficiency

and is well-suited for large-scale problems.

The choice of solver depends on the characteristics of your dataset,

including its size, the presence of regularization, and the specific

logistic regression variant you're using (e.g., binary or multiclass).

Experimenting with different solvers and tuning hyperparameters can help

you find the most effective approach for your specific task.



## **Naive Bayes Classifier**

\- The Naive Bayes classifier is a probabilistic machine learning algorithm

based on Bayes' theorem.

\- It's called "naive" because it makes a strong and often unrealistic

assumption that the features used to describe data points are

conditionally independent, given the class label.

\- The classifier calculates the probability of a data point belonging to

each class and assigns it to the class with the highest probability.

\- Despite its simplicity and the independence assumption, Naive Bayes can

perform surprisingly well in a wide range of classification tasks.

\- It is particularly effective for text classification tasks like spam

email detection and sentiment analysis.

\- Naive Bayes can handle both binary and multiclass classification

problems.

### Confusion Matrix with Naive Bayes:

\- A confusion matrix is a table used to evaluate a Naive Bayes

classifier's performance.

\- It categorizes predictions into true positives, true negatives, false

positives, and false negatives.

-This matrix helps assess model accuracy, precision, recall, and other

performance metrics.

\- It's a vital tool for understanding a Naive Bayes model's strengths and

weaknesses, aiding in model improvement.

# **Regularized Logistic Regression L2- Norm**

\- Regularized Logistic Regression with L2-norm (Ridge Regression) is a

machine learning technique used for classification tasks.

\- It extends traditional Logistic Regression by adding an L2

regularization term to the loss function.

\- L2 regularization helps prevent overfitting by penalizing large

coefficient values.

\- The regularization strength is controlled by a hyperparameter, often

denoted as 'C,' where smaller values indicate stronger regularization.

\- L2 regularization encourages the model to keep all features relevant to

the prediction task.

\- The goal is to find the optimal coefficients that minimize the combined

loss and regularization term.

\- Solver Algorithms in Regularized Logistic Regression:

### SAG (Stochastic Average Gradient) Solver:

SAG is an efficient optimization algorithm for large-scale logistic

regression problems.

It computes gradients using subsets of data, making it suitable for large

datasets.

SAG supports L2 regularization and is known for its speed.

### LIBLINEAR Solver:

LIBLINEAR is a linear solver used primarily for linear classification and

regression.
It works well for both small and large datasets and supports L2-

regularized logistic regression.LIBLINEAR is efficient in terms of memory usage.

### SAGA Solver:

SAGA is an extension of the SAG solver, designed for large-scale and high-

dimensional logistic regression tasks.It supports both L1 and L2 regularization, providing flexibility in

feature selection.SAGA is particularly effective when dealing with a large number of samples

or high-dimensional data.

### Newton-CG (Conjugate Gradient) Solver:

Newton-CG is an optimization algorithm that approximates the Hessian

matrix.\- It can handle both L1 and L2 regularization but may not be as efficient

as SAG for large datasets.

\- Newton-CG is useful when high precision is required in optimization.

\- Confusion matrices, although not directly related to regularization

techniques, are crucial for evaluating the performance of classification

models, including regularized Logistic Regression. They help assess the

model's accuracy, precision, recall, and F1-score by comparing predicted

and actual class labels.

# **Choosing the features**

After choosing Logistic Regression model based on confusion matrix here

where \*\*choose the features\*\* taking in consideration the deployment

phase.

used selectKBest. A built-in feature in sklearn. It picks (I specifically

chose 5 columns) that have the highest correlation with the classification

column. And then I stored them in a separate dataframe and used Logistic

Regression with ‘sag’ solver to predict.so we'll be taking 3 inputs from

user according to code. short-term liability, logarithm of total assets

and constant capital. Then, we made a covariance matrix for that and

converted it into PCA using eigen values and vectors. Then I chose the

number of principal components to keep (1). After that I simply converted

it into numpy array. Applied logistic regression using 'sag' and trained

the model. The model was giving accuracy around 95% and mse around 0.4.

# **Deployment**

you can access our app by following this link [http://ec2-51-20-9-234.eu-

north-1.compute.amazonaws.com:8080/]

## Flask

We also create our app by using flask , then deployed it to AWS . The

files of this part are located into (Deployment\_task.7z) folder. 


### Team Members

\* Habiba Yasser (Team lead)

\* Sara Ashraf

\* Abdulaziz Mustafa

\* Harsh Vaish

\* Rawan Osama

\* Manahil Kamran

\* Sheheryar Sadiq

\* Maliki Ayoub

### Instructor

Yasin Shah

