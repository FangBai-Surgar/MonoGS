import numpy as np
from matplotlib import pyplot as plt
from math import pi
from matplotlib.patches import Ellipse

from numpy import linalg as LA


from matplotlib import style
print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']






def pltCovMat2D (ax, mu, Sigma, edgecolor = 'k',faceColor = 'None' ):
    """
    How to Draw Ellipse of Covariance Matrix
    https://cookierobotics.com/007/
    """
    a = Sigma[0, 0]
    b = Sigma[0, 1]
    c = Sigma[1, 1]
    cont = np.sqrt( ((a-c)/2.0)**2 + b**2 )

    lambda1 = (a+c)/2.0 + cont
    lamdba2 = (a+c)/2.0 - cont
    theta = np.arctan2(lambda1 - a, b) * 180 / np.pi

    width = np.sqrt(lambda1)
    height = np.sqrt(lamdba2)

    ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta,
                        ec=edgecolor, fc=faceColor, lw=2, alpha=0.6)
    ax.add_patch(ellipse)
    
    line_length = 0.2
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    # print(f"eigenvectors: {eigenvectors}")
    eigvec1 = eigenvectors[:, 0]
    eigvec2 = eigenvectors[:, 1]
    vec1_length = line_length * np.sqrt(eigenvalues[0])
    vec2_length = line_length * np.sqrt(eigenvalues[1])
    line1_x = [mu[0] - eigvec1[0] * vec1_length, mu[0] + eigvec1[0] * vec1_length]
    line1_y = [mu[1] - eigvec1[1] * vec1_length, mu[1] + eigvec1[1] * vec1_length]
    
    # Minor axis line
    line2_x = [mu[0] - eigvec2[0] * vec2_length, mu[0] + eigvec2[0] * vec2_length]
    line2_y = [mu[1] - eigvec2[1] * vec2_length, mu[1] + eigvec2[1] * vec2_length]
    
    return ax, np.array([line1_x, line1_y]), np.array([line2_x, line2_y])


def distortByRadial1stOrder (mu, Sigma = None, kappa = 0.01):
    x = mu[0]
    y = mu[1]
    c = 1.0 + kappa*(x*x + y*y)
    if Sigma is not None:
        J = c * np.eye(2) + 2.0 * kappa * np.array([ [x*x,  x*y],  [x*y,  y*y] ])
        return c*mu,  J @ Sigma @ np.transpose(J)  
    else:
        return c*mu

def scaleByFocal (mu, Sigma = None, focal = 1.0):
    if Sigma is not None:
        J = focal * np.eye(2)
        return focal*mu,  J @ Sigma @ np.transpose(J)  
    else:
        return focal*mu


def plotOriginalDistoredEllipses(ax, mu, Sigma, kappa, focal, color_ellipse_original, color_ellipse_distorted):
    mu_dst, Sigma_dst = distortByRadial1stOrder (mu, Sigma, kappa)
    mu_dst_f, Sigma_dst_f = scaleByFocal(mu_dst, Sigma_dst, focal)
    print(f"Original:        mu = {mu}\nSigma = \n{Sigma}")
    print(f"Distorted:       mu = {mu_dst}\nSigma = \n{Sigma_dst}")
    print(f"Distorted_Focal: mu = {mu_dst_f}\nSigma = \n{Sigma_dst_f}")
    if (kappa < 0):
        pltCovMat2D(ax, mu, Sigma, color_ellipse_original, 'snow') 
        pltCovMat2D(ax, mu_dst_f, Sigma_dst_f, None, color_ellipse_distorted)
    else:
        pltCovMat2D(ax, mu_dst_f, Sigma_dst_f, None, color_ellipse_distorted)
        pltCovMat2D(ax, mu, Sigma, color_ellipse_original, 'snow')
    return ax



def main(focal, kappa):

    # command on styles must go first to take effect
    plt.style.use('seaborn-v0_8-talk')
    # plt.style.use('dark_background')
    # plt.style.use('fivethirtyeight')
    # plt.style.use('ggplot')
    # plt.style.use('bmh')


    size = 10
    params = {'legend.fontsize': 'large',
            'figure.figsize': (3.5, 3.5),
            'axes.labelsize': size,
            'axes.titlesize': size*1.3,
            'xtick.labelsize': size*0.75,
            'ytick.labelsize': size*0.75,
            'axes.titlepad': 10,
            }
    
    if focal == 1.0:
        params['xtick.color'] = 'steelblue'
        params['ytick.color'] = 'steelblue'
    else:
        params['xtick.color'] = 'k'
        params['ytick.color'] = 'k'       


    plt.rcParams.update(params)


    dark_red = '#8B0000'
    light_red = '#FF6347'
    light_light_red = '#FFA07A'
    dark_blue = '#00008B'
    light_blue = '#ADD8E6'
    light_light_blue = '#B0E0E6'


    # kappa = -0.08
    # focal = 1.0

    # color_original_points = 'white'
    # color_distorted_points = dark_blue
    color_distortion_field = 'tab:green'

    # color_ellipse_original = light_blue 
    # color_ellipse_distorted = dark_blue


    color_distorted_lines = dark_red
    

    color_ellipse_original = 'tab:blue'
    color_ellipse_distorted = 'tab:orange'

    color1 = 'tab:purple'
    color2 = 'tab:orange'
    color3 = 'tab:blue'
    color4 = 'tab:pink'
    color5 = 'tab:olive'


    nx, ny = (9, 9)
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    xv, yv = np.meshgrid(x, y)

    # 1st order radial distortion
    u = xv * ( kappa * (xv**2 + yv**2) )
    v = yv * ( kappa * (xv**2 + yv**2) )
    xv_new = xv + u
    yv_new = yv + v
    

    fig = plt.figure()
    ax = fig.gca()
    ax.set(xlim=(-focal, focal), ylim=(-focal, focal), aspect="equal")

    ax.fill_between([-1, 1], -1, 1, color='cyan', alpha=0.1)

    # ax.plot(xv, yv, marker='.', color=color_original_points, linestyle='none')
    # ax.plot(xv_new, yv_new, marker='.', color=color_distorted_points, linestyle='none')
    plt.quiver(x,y,u,v, color=color_distortion_field, units='xy', scale=1.0)
    
    ax.plot(xv_new, yv_new, color=color_distorted_lines, linestyle='-', alpha=0.5, lw=0.7) # horizontal lines
    ax.plot(np.transpose(xv_new), np.transpose(yv_new), color=color_distorted_lines, linestyle='-', alpha=0.5, lw=0.7) # vertical lines
    

    # ellipse 1
    mu = np.array([0.5, 0.5])
    Sigma = 0.5*np.array([ [1, 0.5], [0.5, 1.5] ])
    Sigma = Sigma @ Sigma
    plotOriginalDistoredEllipses(ax, mu, Sigma, kappa, focal, color1, color1)

    # ellipse 2
    mu = np.array([-0.5, -0.65])
    Sigma = 0.5*np.array([ [1.5, -0.5], [-0.5, 0.5] ])
    Sigma = Sigma @ Sigma
    plotOriginalDistoredEllipses(ax, mu, Sigma, kappa, focal, color2, color2)

    # ellipse 3
    mu = np.array([-0.5, 0.5])
    Sigma = 0.5*np.array([ [1.5, 0], [0, 0.5] ])
    Sigma = Sigma @ Sigma
    plotOriginalDistoredEllipses(ax, mu, Sigma, kappa, focal, color3, color3)

    # ellipse 4
    mu = np.array([0.7, -0.5])
    Sigma = 0.5*np.array([ [0.5, 0], [0, 1.5] ])
    Sigma = Sigma @ Sigma
    plotOriginalDistoredEllipses(ax, mu, Sigma, kappa, focal, color4, color4)

    # ellipse 5
    mu = np.array([-0., 0.])
    Sigma = 0.5*np.array([ [1.5, -0.5], [-0.5, 0.5] ])
    Sigma = Sigma @ Sigma
    plotOriginalDistoredEllipses(ax, mu, Sigma, kappa, focal, color5, color5)

    ax.set_title(f'$\kappa = {kappa}$,   $f = {focal}$')

    distortion_type = "barrel" if kappa < 0 else "pincushion"

    plt.tight_layout(pad = 0.5)
    plt.savefig(f"gaussian_distortion_{distortion_type}_{focal}.pdf")    
    plt.show(block=True)




if __name__ == "__main__":

    main(focal = 1.0, kappa = -0.08)
    main(focal = 1.0, kappa = 0.08)

    main(focal = 2.0, kappa = -0.08)
    main(focal = 2.0, kappa = 0.08)
