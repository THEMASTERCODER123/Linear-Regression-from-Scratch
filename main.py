import random
import matplotlib.pyplot as plt
import numpy as np

def func_x(x):
    return ((4*x) + 3) # |y = mx + b| ~ |m = 4 & b = 3|

def noisy_gen_XY(equation, noOfPairs):
    list_of_pair_xy = []
    x_list = []
    y_list = []
    cntr = 0
    for i in range(noOfPairs):
        x = cntr
        y = equation(x)
        noisy_x = x + random.uniform(random.random(), 1)
        noisy_y = y + random.uniform(random.random(), 1)
        list_of_pair_xy.append([noisy_x, noisy_y])
        x_list.append(noisy_x)
        y_list.append(noisy_y)
        
        cntr += 1
    #return list_of_pair_xy, x_list, y_list
    #list_of_pair_xy = genXY(func_x, 10)[0]
    #x_list = genXY(func_x, 10)[1]
    #y_list = genXY(func_x, 10)[2]
    return x_list, y_list

def plain_gen_XY(equation, noOfPairs):
    list_of_pair_xy = []
    x_list = []
    y_list = []
    cntr = 0
    for i in range(noOfPairs):
        x = cntr
        y = equation(x)
        list_of_pair_xy.append([x, y])
        x_list.append(x)
        y_list.append(y)
        
        cntr += 1
    #return list_of_pair_xy, x_list, y_list
    #list_of_pair_xy = genXY(func_x, 10)[0]
    #x_list = genXY(func_x, 10)[1]
    #y_list = genXY(func_x, 10)[2]
    return x_list, y_list

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

def plot(x_list, y_list, function):
    plt.ion()

    fig, ax = plt.subplots()

    # Random/noisy data points
    points = ax.scatter(
        x_list,
        y_list,
        label="Random training points"
    )

    # X values used for drawing both lines
    x_line = np.linspace(
        min(x_list),
        max(x_list),
        100
    )

    # Actual function: y = 4x + 3
    actual_y = []

    for x in x_line:
        actual_y.append(function(x))

    ax.plot(
        x_line,
        actual_y,
        linestyle="--",
        label="Actual function"
    )

    # Initial learned line: m = 0, b = 0
    learned_line, = ax.plot(
        x_line,
        0 * x_line,
        label="Learned best-fit line"
    )

    ax.set_title("Linear Regression Training")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":")
    ax.legend()

    fig.canvas.draw()
    fig.canvas.flush_events()

    # Give you time to initially see the points/function
    plt.pause(1)

    return fig, points, learned_line, x_line

def run():
    #x_list = gen_XY(func_x, 10)[1]
    #y_list = gen_XY(func_x, 10)[2]
    x_list, y_list = noisy_gen_XY(func_x, 10)

    m = 0 #initial guess
    b = 0  #initial guess

    learning_rate = 0.01

    epochs = 5000
    epochs_passed = 0
    fig, points, learned_line, x_line = plot(x_list, y_list, func_x)
    for i in range(epochs):
        predictions = genPredictions(x_list, m, b)
        current_loss = loss_mse(predictions, y_list)
        m_gradient, b_gradient = gradients(x_list, y_list, predictions)
        m = m - (learning_rate*m_gradient)
        b = b - (learning_rate*b_gradient)
        

        if i % 10 == 0:
            points.set_offsets(np.column_stack((x_list, y_list)))
            learned_line.set_ydata(m * x_line + b)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)
            print(f'Epoch: {epochs_passed}  |  Loss MSE: {current_loss:.30f}')
        epochs_passed += 1
    print(f'Final | m: {m:.20f} | b: {b:.20f} |')

if __name__ == "__main__":
    run()

