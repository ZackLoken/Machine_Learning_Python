# Zack Loken CSC 7333 Module 2 Assignment - Part A
# July 17, 2022

# Import necessary Python libraries etc. 
import numpy as np # for linear algebra
import pandas as pd # for data processing
import matplotlib.pyplot as plt # for data visualization

##-------------------------------------------------------------------------------------------------##
##                          Assignment 2 - Part A: Single Input Variable                           ##
##-------------------------------------------------------------------------------------------------##

# Evaluate the linear regression
# using the mean squared error (MSE) loss function
def loss_function(x, y, theta):
    '''
    Compute cost (loss) for linear regression
    using mean squared error (MSE)
    '''
    m = float(len(x)) # training sample size
    predictions = x.dot(theta).flatten() # the dot product of theta with x**i (i.e., predicted value)
    sqError = (predictions - y) ** 2 # dot product of theta with x**i minus y
    j = (1.0/(2 * m)) * sqError.sum() # take the sum of squared error and multiply with the inverse of (2*m) to find loss (J)
    
    return j # the MSE loss function

# and the gradient descent algorithm
def gradient_descent(x, y, theta, lr, epochs):
    '''
    Uses gradient descent to learn theta from
    number of epochs and the learning rate
    '''
    m = float(len(x)) # training sample size
    past_loss = np.zeros(shape = (epochs, 1)) # define shape of array for storing loss values

    for n in range(epochs):
        predictions = x.dot(theta).flatten() # the dot product of theta with x**i
        errors_x1 = (predictions - y) * x[:, 0] # initialize function for storing intercept errors
        errors_x2 = (predictions - y) * x[:, 1] # initialize function for storing slope errors
        theta[0] = theta[0] - lr * (1.0 / m) * errors_x1.sum() # theta 0 is predicted intercept using gradient descent
        theta[1] = theta[1] - lr * (1.0 / m) * errors_x2.sum() # theta 1 is predicted slope using gradient descent
        past_loss[n, 0] = loss_function(x, y, theta) # calculated loss values at each epochal step n
    
    return theta, past_loss

# Read in the single predictor variable dataset
data_a = pd.read_csv('KCSmall2.csv', header = None) # load the .csv file (no headers)
x = data_a.iloc[:, 0] # specify that x values are in first column
y = data_a.iloc[:, 1] # specify that y values are in second column

# Define parameters for plot size
plt.rcParams['figure.figsize'] = (11.5, 8.5)

# Scatter plot to confirm data loaded correctly [Part A, Question 1]
plt.scatter(x, y, marker = "x", label = "Training Data") # scatter plot of training data
plt.title("Home Values by Square Footage") # plot title
plt.xlabel("House Square Footage in 1,000 Sq. Ft.") # x-axis label
plt.ylabel("House Cost in $10,000 Dollars") # y-axis label
plt.legend(frameon=False, loc = 'best') # plot legend
plt.show() # display plot

# Variable for number of training samples
m = x.size

# Add a column of ones in the input data to accommodate the
# bias term while finding the optimal hypothesis function
b = np.ones(shape = (m, 2)) # add another dimenson filled with ones
b[:, 1] = x # to the input to intercept term

# Initialize the theta parameters for fitting
theta = np.zeros(shape = (2, 1)) # theta = (0, 0) in initial state

# Calculate and print inital cost (loss) and cost when theta = (-1, 20) [Part A, Question 2]
theta = [0, 0] # Set theta to it's inital state, theta = (0,0)
print(f"The loss function value when theta = (0, 0) is {loss_function(b, y, theta):.3f}") # loss value when theta = (0, 0)

theta = [-1, 20] # check MSE when theta = (-1, 20)
print(f"The loss function value when theta = (-1, 20) is {loss_function(b, y, theta):.3f}") # loss value when theta = (-1, 20)
print() # print empty line in terminal

# Gradient descent hyperparameters
epochs = 15 # number of iterations to train model
learning_rate = [0.01, 0.1, 0.2, 0.4] # different learning rates to train across

# Pass the relevant variables to the function and get new values back after running gradient descent
for lr in learning_rate:
    theta = [0, 0] # initialize theta to initial state [0, 0]
    theta, past_loss = gradient_descent(b, y, theta, lr, epochs) # store updated thetas and loss values after gradient descent
    
# Plot loss function curve across four trials (one for each learning rate) [Part A, Question 3]
    plt.figure()
    plt.plot(past_loss, label = f"Loss Curve for Learning Rate {lr}") # plot loss curve 
    plt.title(f"Loss Function Value (J) for Learning Rate {lr}") # plot title w/ learning rate value added
    plt.xlabel("Epochs") # x-axis label
    plt.ylabel("Loss (J)") # y-axis label
    plt.legend(frameon=False, loc = 'best') # plot legend
    plt.show()

# Final theta vector learned and associated loss value (J) across each of the four trials [Part A, Question 4]
    print(f"The final theta vector learned using a learning rate of {lr} is (%.3f, %.3f)" % (theta[0], theta[1])) # final theta vector
    print(f"The associated loss value for the final theta vector using a learning rate of {lr} is %.3f" % (past_loss[-1])) # associated loss value
    print() # print empty line in terminal

# Predicted y values for x = 3.5 (3,500 sq. ft.) and x = 7.0 (7,000 sq. ft.) across each of the four trials [Part A, Question 4]
    y_hat3500 = np.array([1, 3.5]).dot(theta).flatten() # when x = 3.5
    print(f"When x = 3.5, the predicted y = %.3f using a learning rate of {lr}" % (y_hat3500))
    print(f"The predicted cost of a 3,500 sq. ft. home is $%.2f using a learning rate of {lr}" % (y_hat3500 * 10000))
    print() # print empty line in terminal

    y_hat7000 = np.array([1, 7.0]).dot(theta).flatten() # when x = 7.0
    print(f"When x = 7.0, the predicted y = %.3f using a learning rate of {lr}" % (y_hat7000))
    print(f"The predicted cost of a 7,000 sq. ft. home is $%.2f using a learning rate of {lr}" % (y_hat7000 * 10000))
    print() # print empty line in terminal
