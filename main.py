def func_x(x):
    return ((4*x) + 3) # |y = mx + b| ~ |m = 4 & b = 3|

def gen_XY(equation, noOfPairs):
    list_of_pair_xy = []
    x_list = []
    y_list = []
    cntr = 0
    for i in range(noOfPairs):
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

def loss(prediction, true_y):
    total_error = 0

def run():
    x_list = gen_XY(func_x, 10)[1]
    y_list = gen_XY(func_x, 10)[2]

    m = 0 #initial guess
    b = 0 #initial guess

    learning_rate = 0.001

    epochs = 1000
    print("done")

if __name__ == "__main__":
    run()

