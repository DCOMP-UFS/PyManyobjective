from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def plot(points, title="", three_dim=False):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    fig = plt.figure(figsize=(8, 6))

    if three_dim:
        z_coords = [p[2] for p in points]
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=45, azim=45)
        ax.scatter(x_coords, y_coords, z_coords, c='#0073CF')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(x_coords, y_coords, c='#0073CF')
    
    ax.set_title(title)
    plt.show()

def plot_from_file(file_name):
    with open(f'results/{file_name}/F0', 'r') as file:
        content = file.read()
    points = eval(content)
    three_dim = False
    if len(points[0]) == 3:
        three_dim = True
    plot(points, file_name, three_dim=three_dim)

plot_from_file('M6_DTLZ2_NSGAII')