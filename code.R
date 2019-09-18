---
title: |
    | Optimization Techniques 
    | Linear and Logistic Regression
author: "James Ponce"
date: "March 5, 2019"
output:
  pdf_document: default
  html_document:
    df_print: paged
bibliography: references.bib
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

##Introduction

In this study we are going to explain the basis of optimization techniques for two models: the linear regression model and the logistic regression model. For both of these we will cover the implementation of the models in R, after that we will compute the models to show the numeric results. Finally, we will make a result comparison between the model made from scratch in contrast with the results obtained using the R build-in funcions, such as lm (for linear regression) and glm(for logistic regression).


##Linear Regression Model
  
$$ $$  
  
**Model**


The following equation represents the linear model in matrix notation:  

$$\Large y=\beta X+\epsilon$$

Where:

$X$: is the matrix that represents the independent variables.  
$y$: is the vector that represents the dependent or response variable.  
$\beta$: is the vector of coefficients.  
$\epsilon$: is the vector of errors.  

The aim of this model is minimize the square sum of errors, represented by:  

$$\sum_{i = 1}^{n}\epsilon^{2}=
\left[\begin{array}{r}\epsilon_1 \epsilon_2 \epsilon_3,...\epsilon_n \end{array}\right]
\left[\begin{array}
{rrrr}
\epsilon_1 \\
\epsilon_2 \\
\epsilon_3
\end{array}\right]=\epsilon\epsilon^{t}$$

Replacing $\epsilon= y-x\beta$ in the previous equation, we have:


$$\Large \epsilon\epsilon^{t}=(y-X\beta)^{t}(y-X\beta)$$

We calculate the first derivative 

$$\Large \frac{\partial(\epsilon\epsilon^{t})}{\partial \beta}=\frac{\partial((y-X\beta)^{t}(y-X\beta))}{\partial \beta}$$

$$\Large \frac{\partial(\epsilon\epsilon^{t})}{\partial \beta}=-2X^{t}(y-X\beta)$$

\newpage

Given the theory, we make the previous function equal to zero, then solve for $\beta$.  

$$\Large -2X^{t}(y-X\beta)=0$$  

Or assuming the normal equation: 

$$\Large X^{t}y=X^{t}X\beta$$  

The solution for obtaining the coefficients, vector $\beta$ is:  

$$\Large \beta=(X^{t}X)^{-1}X^ty$$

Where:  


$X^{t}$: is the transpose matrix of $X$.    
$X^{-1}$: is the inverse matrix of $X$.  
  
$$ $$  
  
**Data**


In order to test this model we use the dataset Boston which is included in the MASS R package.
The Boston data frame includes 14 variables 506 observations. Among those variables we consider one dependent variable which is the `median value of owner-occupied homes` and 13 independent variables such as `crime rate, tax rate, number of rooms, etc`.
  
   
[[Boston Dataset](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/Boston.html)]


```{r}
library(MASS)
str(Boston)
```
  
$$ $$  
  
**Implementation in R**


According to the model in matrix notation, we implement the calculation of the coefficients vector $\beta$ as follows using matrix operations.


```{r}
# library(MASS)

# Dependent variable median value of owner-occupied homes
y <- Boston$medv

# Matrix of feature variables from Boston
# Select all variables but the last "medv"
X <- as.matrix(Boston[-ncol(Boston)])

# Vector of ones with same length as rows in Boston
int <- rep(1, length(y))

# Add intercept column to X
X <- cbind(int, X)

# Implement closed-form solution
# Solve function compute the inverse of matrix [t(X)%*%X]
betas <- solve(t(X) %*% X) %*% t(X) %*% y

# Round with 2 decimals
betas <- round(betas, 2)
```
  
$$ $$  
  
**Numeric Results**

We compare the results that we have obtained using our function in contrast with the results computed with `lm()` function. The results are similar, its mean that we have the same $\beta$ coefficientes for the optimized objective function and, of course, the same prediction and error.

Also, we can consider the variable `int` which means `proportion of non-retail business acres per town` has the most positive impact in the `median value of the house`. In the other hand, we realized that the variable `nox` which means `nitrogen oxides concentration` represents the most negative effect to our response variable.


```{r}
# Comparison face to lm() function
# lm for linear regression model
lm.mod <- lm(medv ~ ., data=Boston)

# Round with 2 decimals
lm.betas <- round(lm.mod$coefficients, 2)

# Create data.frame of results
results <- data.frame(own_function=betas, lm_results=lm.betas)
print(results)
```
  
$$ $$  
  
##Logistic Regression Model
  
$$ $$  
  
**Model**

The following equation represents the logistic regression model in matrix notation:   

$$\Large \log(\frac{p}{(1-p)}) = \theta_0+\theta_1X_1+\theta_2X_2+\theta_3X_3...+\theta_nX_n$$  

The cost function is represented by the following equation:  

$$\Large J = \frac{-1}{m} \sum_{i = 1}^{m} (y^{(i)}log(h(x^{(i)})) + (1-y^{(i)})log(1-h(x^{(i)}))$$ 

Where $h(x)$ is the sigmoid function, defined by:  

$$\Large h(x) = \frac{1}{1+e^{-x.\theta}}$$  
   
$$ $$  
  
**Data**

For testing this logistic regression model, we use the dataset `NBA stats` to predict an outcome (made or not) of a shot based on the shot distance, shot clock remaining and distance to closest defender.

The `NBA stats` data frame used for this study includes 4 variables 203590 observations. Among those variables we consider one dependent variable `FGM` which is the `shot made` and 3 independent variables such as `shot clock remaining, shot distance and distance to closest defender`.

[[NBA stats Dataset](https://github.com/JunWorks/NBAstat)]
```{r}
# Load the dataset
# library(dplyr)
shot <- read.csv('shot.csv', header = T, stringsAsFactors = F)
shot.df <- shot[,c("FGM", "SHOT_CLOCK", "SHOT_DIST", "CLOSE_DEF_DIST")]
str(shot.df)
```
 
**Implementation in R**

According to the model in matrix notation, we implement three functions: the sigmoid, which is the inverse of the logit function, then we use that function to calculate the cost function. After that, we implement the gradient function. Finally, we compute the logistic regression function that uses the cost function and the gradient function into the optim function to optimize the response variable $y$.

```{r message=FALSE, warning=FALSE}
# Loading package
library(dplyr)

# Sigmoid function, inverse of logit
sigmoid <- function(z){1/(1+exp(-z))}

# Cost function
cost <- function(theta, X, y){
  m <- length(y) # number of training examples
  
  h <- sigmoid(X%*%theta)
  J <- (t(-y)%*%log(h)-t(1-y)%*%log(1-h))/m
  J
}

# Gradient function
grad <- function(theta, X, y){
  m <- length(y) 
  
  h <- sigmoid(X%*%theta)
  grad <- (t(X)%*%(h - y))/m
  grad
}

logisticReg <- function(X, y){
  # Remove NA rows
  temp <- na.omit(cbind(y, X))
  # Add bias term and convert to matrix
  X <- mutate(temp[, -1], bias =1)
  X <- as.matrix(X[,c(ncol(X), 1:(ncol(X)-1))])
  y <- as.matrix(temp[, 1])
  # Initialize theta
  theta <- matrix(rep(0, ncol(X)), nrow = ncol(X))
  # Use the optim function to perform gradient descent
  costOpti <- optim(matrix(rep(0, 4), nrow = 4), cost, grad, X=X, y=y)
  # Return coefficients
  return(costOpti$par)
}
```
  
$$ $$  
  
**Numeric Results**

We compare the results that we have obtained using our function in contrast with the results computed with `glm()` function. The results are similar, its mean that we have the same $\beta$ coefficientes for the optimized objective function.


```{r}
# Computing own logisticReg function to calculate coeficients
# Dividing dataset in variables X, y
shot.X <- shot.df[, -1]
shot.y <- shot.df[, 1]

mod <- logisticReg(shot.X, shot.y)
own_function<-as.vector(mod)

# Using glm
mod1 <- glm(as.factor(FGM) ~ SHOT_CLOCK + SHOT_DIST + CLOSE_DEF_DIST, 
            family=binomial(link = "logit"), data=shot.df)
lm_results<-as.vector(mod1$coefficients)

# Comparing with lm() function
comparison<-cbind(own_function,lm_results)
rownames(comparison)<-c("(intercept)","SHOT_CLOCK","SHOT_DIST","CLOSE_DEF_DIST")
comparison
```
  
$$ $$  
  
##References  

[@MultipleRegressionMatrix]
[@MassHousingDataset]
[@MatrixDifferentiation]
[@LogisticRegression]
[@NBAstats]



