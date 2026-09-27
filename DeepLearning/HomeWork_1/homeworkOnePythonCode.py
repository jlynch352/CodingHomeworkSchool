import matplotlib.pyplot as plt
import numpy as np

def Function(x, y):
    return x**2 + y**2

def gradient(x, y):
    return 2*x, 2*y

# Starting point
x = 10.0
y = 10.0

# Learning rate
learning_rate = 0.1

# Number of SGD iterations
num_iterations = 100

#Records 
x_history = [x]
y_history = [y]
z_history = [Function(x, y)]

print("Initial point: ", (x, y), "Function: ", Function(x, y))

# Gradient Descent Loop
for i in range(1, num_iterations + 1):
    #update x and y using the gradient
    grad_x, grad_y = gradient(x, y)
    x = x - learning_rate * grad_x
    y = y - learning_rate * grad_y

    # Save history for plotting
    x_history.append(x)
    y_history.append(y)
    z_history.append(Function(x, y))


#print results to cli
print("\nFinal solution:")
print("x =", x)
print("y =", y)
print("Minimum value =", Function(x, y))


#Set up Figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

#adds actual function to graph
grid = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(grid, grid)
ax.plot_surface(X, Y, Function(X, Y), color="green", alpha=0.4)
ax.plot(x_history, y_history, z_history, color="red", marker="o")

#Labels and Text
ax.set_title("f(x, y) = x² + y²")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.text2D(0.05, 0.95, f"Learning Rate: {learning_rate}", transform=ax.transAxes)


plt.show()







