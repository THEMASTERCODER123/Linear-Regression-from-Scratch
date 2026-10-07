def function_x(x):
    return ((4*x) + 3) # |y = mx + b| ~ |m = 4 & b = 3|

def genXY(equation, noOfPairs):
    list_of_pair_xy = []
    cntr = 0
    for i in range(noOfPairs):
        x = cntr
        y = equation(x)
        list_of_pair_xy.append((x,y))
        cntr += 1
    return list_of_pair_xy

#print(genXY(function_x, 500))

