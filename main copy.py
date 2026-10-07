import numpy as np

def complex_linear_regression_test_func(x):
    return (14.5 
            - 3.2 * x[0] 
            + 0.85 * (x[1]**2) 
            + 12.1 * np.log(x[2]) 
            - 0.45 * (x[0] * x[1]) 
            + 5.6 * np.sin(x[3]) 
            - 22.3 * np.exp(-x[4]))

def func_x(x):
    return ((4*x) + 3) # |y = mx + b| ~ |m = 4 & b = 3|

def gen_XY(equation, noOfPairs):
    list_of_pair_xy = []
    x_list = []
    y_list = []
    cntr = 0
    for i in range(noOfPairs):
        if equation == complex_linear_regression_test_func:
            x = np.array([
                cntr + 1,
                cntr + 1,
                cntr + 1,
                cntr + 1,
                cntr + 1
            ], dtype=float)
        else:
            x = cntr

        y = equation(x)
        list_of_pair_xy.append((x,y))
        x_list.append(x)
        y_list.append(y)
        cntr += 1
    return list_of_pair_xy, x_list, y_list
    #list_of_pair_xy = genXY(func_x, 10)[0]
    #x_list = genXY(func_x, 10)[1]
    #y_list = genXY(func_x, 10)[2]

def predict(x, m, b):
    y = m*x + b
    return y 

def loss_mse(predictions, true_y):
    total_error = 0

    for i in range(len(true_y)):
        error = true_y[i] - predictions[i]
        sq_error = error**2
        total_error += sq_error
    mse = total_error/len(true_y)
    return mse

def genPredictions(x_list, m, b):
    predictions= []
    for x in x_list:
        predictions.append(predict(x,m,b))
    return predictions

def gradients(x_list, y_list, predictions):
    m_gradient = 0
    b_gradient = 0

    for i in range(len(x_list)):
        error = y_list[i] - predictions[i]

        m_gradient += -2 * x_list[i] * error
        b_gradient += -2 * error

    m_gradient /= len(x_list)
    b_gradient /= len(x_list)

    return m_gradient, b_gradient

def run():
    x_list = gen_XY(complex_linear_regression_test_func, 10)[1]
    y_list = gen_XY(complex_linear_regression_test_func, 10)[2]

    m = 0 #initial guess
    b = 0 #initial guess

    learning_rate = 0.01

    epochs = 10000
    epochs_passed = 0
    for i in range(epochs):
        predictions = genPredictions(x_list, m, b)
        current_loss = loss_mse(predictions, y_list)
        gradients_xy = gradients(x_list, y_list, predictions)
        m_gradient, b_gradient = gradients_xy
        m = m - (learning_rate*m_gradient)
        b = b - (learning_rate*b_gradient)
        epochs_passed += 1
        
        print(f'Epoch: {epochs_passed}  |  Loss MSE: {current_loss}')

    print(f'Final | m: {m} | b: {b} |')


    


if __name__ == "__main__":
    run()

