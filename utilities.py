import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bloch_axes(ax_width=0.8):
    ''' Plots the basic axis for a Bloch Sphere plot. '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.set_aspect('equal')
    
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    
    # Plot Sphere
    ax.plot_surface(x, y, z,  rstride=5, cstride=5, linewidth=1, alpha=0.2, color='grey')
    ax.plot_surface(x, y, z - z, alpha=0.2, color='grey')
    
    # Plot Axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='k', linewidth=ax_width)
    ax.plot([0, 0], [-1, 1], [0, 0], color='k', linewidth=ax_width)
    ax.plot([0, 0], [0, 0], [-1, 1], color='k', linewidth=ax_width)
    
    # Plot Labels
    ax.text(0, 0, 1.1, '|0⟩')
    ax.text(0, 0, -1.2, '|1⟩')
    ax.text(1.1, 0, 0, 'x')
    ax.text(0, 1.1, 0, 'y')
    
    return fig, ax

def bloch_vector(alpha, beta):
    ''' Takes the alpha and beta values of a qubit state vector and returns the 3D Bloch Vector. '''
    
    if alpha.real < 0 and alpha.imag > 0 :
        theta_a = np.arctan(alpha.imag / alpha.real) + np.pi
    elif alpha.real < 0 and alpha.imag < 0 :
        theta_a = np.arctan(alpha.imag / alpha.real) + np.pi
    else:
        theta_a = 0 if alpha.real == 0 else np.arctan(alpha.imag / alpha.real) 
    
    r_a = np.sqrt((alpha.real**2) + (alpha.imag**2))

    if beta.real < 0 and beta.imag > 0 :
        theta_b = np.arctan(beta.imag / beta.real) + np.pi
    elif beta.real < 0 and beta.imag < 0 :
        theta_b = np.arctan(beta.imag / beta.real) + np.pi
    else:
        theta_b = 0 if beta.real == 0 else np.arctan(beta.imag / beta.real) 
    
    phi = theta_b - theta_a
    theta = 2 * np.arccos(r_a)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.array([x, y, z])
