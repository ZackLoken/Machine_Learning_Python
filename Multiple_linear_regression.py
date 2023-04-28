# Zack Loken CSC 7333 Module 2 Assignment - Part B
# July 17, 2022

# Import necessary Python libraries etc. 
import numpy as np # for linear algebra
import pandas as pd # for data processing
import matplotlib.pyplot as plt # for data visualization

#-------------------------------------------------------------------------------------------------##
#                         Assignment 2 - Part B: Multiple Input Variables                         ##
#-------------------------------------------------------------------------------------------------##

# Standardize the input variables because their ranges differ significantly 
def standardize_input(x):
    '''
    Returns standardized x values
    by subtracting mean (x) from x and 
    diving by standard deviation (x)
    '''
    x_mu = np.mean(x, axis = 0) # mean (mu) of each x
    x_sigma = np.std(x, axis = 0, ddof = 0) # standard deviation (sigma) of each x 
    x_norm = (x - x_mu)/x_sigma # subtract mean value x from x and divide by standard deviation

    return x_norm, x_mu, x_sigma

# Evaluate the multiple linear regression
# using the mean squared error (MSE) loss function
def loss_function(x, y, theta):
    '''
    Compute cost (loss) for linear regression
    using the mean squared error (MSE)
    '''
    m = float(len(x)) # training sample size
    predictions = x.dot(theta).flatten() # the dot product of theta with x**i (i.e., predicted value)
    sqError = (predictions - y) ** 2 # predicted value minus true value
    j = (1.0/(2 * m)) * sqError.sum() # transpose error matrix, take dot product with itself, and multiply by the inverse of (2*m) to find loss (J)
    
    return j # the MSE loss function

def gradient_descent(x, y, theta, lr, epochs):
    '''
    Uses gradient descent to learn theta from
    number of epochs and the learning rate
    '''
    m = float(len(x)) # training sample size
    past_loss = np.zeros(shape = (epochs, 1)) # define array shape for storing loss values

    for n in range(epochs):
        predictions = x.dot(theta).flatten() # the dot product of theta with x**i
        errors_x1 = (predictions - y) * x[:, 0] # initialize function for storing intercept errors
        errors_x2 = (predictions - y) * x[:, 1] # initialize function for storing 1st slope errors
        errors_x3 = (predictions - y) * x[:, 2] # initialize function for storing 2nd slope errors
        errors_x4 = (predictions - y) * x[:, 3] # initialize function for storing 3rd slope errors
        theta[0] = theta[0] - lr * (1.0 / m) * errors_x1.sum() # predicted intercept using gradient descent
        theta[1] = theta[1] - lr * (1.0 / m) * errors_x2.sum() # 1st predicted slope using gradient descent
        theta[2] = theta[2] - lr * (1.0 / m) * errors_x3.sum() # 2nd predicted slope using gradient descent
        theta[3] = theta[3] - lr * (1.0 / m) * errors_x4.sum() # 3rd predicted slope using gradient descent 
        past_loss[n, 0] = loss_function(x, y, theta) # calculated loss values at each epochal step n
    
    return theta, past_loss

# Read in the multivariate dataset
data_b = pd.read_csv('KCSmall_NS2.csv', header = None) # load the .csv file (no headers)
x = data_b.iloc[:, :3] # input variables from columns 0, 1, and 2
y = data_b.iloc[:, 3] # output variables from column 3

# Define parameters for plot size
plt.rcParams['figure.figsize'] = (11.5, 8.5)

# Number of training samples
m = len(x)

# Print out the first 5 rows of raw input data and check that m = 100 [Part B, Question 1]
data_bF5 = data_b.head(n = 5) # object with first 5 rows (n = 5)
print(f"The first 5 rows of the raw input data are: {data_bF5}") # print it 
print() # print empty line in terminal

print(f"The number of input rows in the training data (m) = {m}") # checking that m = 100
print() # print empty line in terminal

# Standardize the training data
x, x_mu, x_sigma = standardize_input(x) # Store normalized values

# Add a column of ones to the input data (i.e., 0th dummy column)
b = np.ones(shape = (m, 4))
b[:, 1:4] = x

# Print out the first 5 rows of normalized data after adding the dummy 0th column [Part B, Question 2]
xF5 = x[:5] # first 5 rows
print(f"The normalized values for the first 5 rows after adding the dummy 0th column are: {xF5}")
print() # print empty line in terminal

# Define additional hyperparameters for gradient descent
epochs = 50 # number of iterations
learning_rate = [0.01, 0.1, 0.5, 1.0, 1.5] # learning rates to test when training predictive model

# Initialize theta parameters for fitting
theta = np.zeros(shape = (4, 1))

# Calculate and print the loss function value when theta is in it's initial state theta = (0, 0, 0, 0) [Part B, Question 3]
theta = [0, 0, 0, 0] # Set theta to it's inital state, theta = (0, 0, 0, 0)
print(f"The loss function value when theta = (0, 0, 0, 0) is {loss_function(b, y, theta):.3f} or {loss_function(b, y, theta):.3e}") # loss when theta = (0, 0, 0, 0)
print() # print empty line in terminal

# Pass the relevant variables to the function and get new values back after running gradient descent
for lr in learning_rate:
    theta = [0, 0, 0, 0] # initialize theta to initial state [0, 0, 0, 0]
    theta, past_loss = gradient_descent(b, y, theta, lr, epochs) # store updated thetas and loss values after gradient descent
    
# Plot loss function curve across the five trials (one for each learning rate) [Part B, Question 4]
    plt.figure()
    plt.plot(past_loss, label = f"Loss Curve for Learning Rate {lr}") # plot loss curve 
    plt.title(f"Loss Function Value (J) for Learning Rate {lr}") # plot title w/ learning rate value added
    plt.xlabel("Epochs") # x-axis label
    plt.ylabel("Loss (J)") # y-axis label
    plt.legend(frameon=False, loc = 'best') # plot legend
    plt.show()

# Final theta vector learned and associated loss value (J) across each of the five trials [Part B, Question 5]
    print(f"The final theta vector learned using a learning rate of {lr} is (%.3f, %.3f, %.3f, %.3f)" % (theta[0], theta[1], theta[2], theta[3])) # final theta vector
    print(f"The associated loss value for the final theta vector using a learning rate of {lr} is %.3f or %.3e" % (past_loss[-1], past_loss[-1])) # associated loss value
    print() # print empty line in terminal

# Predicted y value when n_bed = 3, liv_area = 2000, lot_area = 8550 across each of the five trials [Part B, Question 5]
    # input values standardized to match training data
    y_hat = np.array([1, ((3 - x_mu[0]) / x_sigma[0]), ((2000 - x_mu[1]) / x_sigma[1]), ((8550 - x_mu[2]) / x_sigma[2])]).dot(theta).flatten() # standardized values substituted
    print(f"When n_bed = 3, liv_area = 2000, and lot_area = 8550, the predicted y = %.3f using a learning rate of {lr}" % (y_hat))
    print(f"The predicted cost of a house with 3 bedrooms, living area size of 2,000 and lot area size of 8,550 is $%.2f using a learning rate of {lr}" % (y_hat))
    print()
