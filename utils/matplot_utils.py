
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.text import OffsetFrom

# from mpl_toolkits.axes_grid.axes_grid import AxesGrid
# from mpl_toolkits.axes_grid.anchored_artists import AnchoredText





def annotate_image(image, cmap=plt.get_cmap('hot'), mytext = "my text", mytextt = None, mytexttt = None):
    """
    Annotate an image with text on the upper-left corner
    """
    h, w, _ = image.shape
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    fig = plt.figure(figsize=(w*px, h*px))
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.axis('off')
    fig.add_axes(ax)
    '''
    need to specify axis range explicitly
    https://stackoverflow.com/questions/13018115/matplotlib-savefig-image-size-with-bbox-inches-tight
    '''
    im = ax.imshow(image, cmap=cmap, interpolation='nearest')
    
    '''
    Annotations
    https://matplotlib.org/1.5.3/users/annotations_guide.html   
    '''
    padding = 5
    offset_x, offset_y = -2, -2
    background = '#343837'   #'#363737'
    '''
    xkcd_fig = plot_colortable(mcolors.XKCD_COLORS)
    xkcd_fig.savefig("XKCD_Colors.png")
    https://xkcd.com/color/rgb/
    https://matplotlib.org/stable/gallery/color/named_colors.html
    '''

    boxprops = dict(fill = True, facecolor=background, edgecolor=background, pad=padding)

    """
    paddings
    https://stackoverflow.com/questions/38480739/aligning-a-text-box-edge-with-an-image-corner/38487750
    """
    if mytext is not None:
        an1 = ax.annotate(
            text = mytext,
            fontsize = 20,
            color='snow',
            xy=(0, 0),
            xytext=(padding-offset_x, -(padding-offset_y)),
            textcoords = 'offset pixels',
            bbox=boxprops,
            va='top',
            ha='left',
            clip_on = True,
            )

    if mytextt is not None:
        an2 = ax.annotate(
            text = mytextt,
            fontsize = 45,
            color='red',
            xy=(w/2, h/2),
            xytext=(padding-offset_x, -(padding-offset_y)),
            textcoords = 'offset pixels',
            # bbox=boxprops,
            va='center',
            ha='center',
            clip_on = True,
            )    

    if mytexttt is not None:
        an3 = ax.annotate(
            text = mytexttt,
            fontsize = 20,
            color='snow',
            xy=(w/2, h-50),
            xytext=(padding-offset_x, -(padding-offset_y)),
            textcoords = 'offset pixels',
            # bbox=boxprops,
            va='top',
            ha='center',
            clip_on = True,
            )

    return fig, ax, im

 

 





def annotate_image_by_table (image, cmap=plt.get_cmap('hot'), mytext = "my text"):
    """
    Annotate an image with text on the upper-left corner
    """
    h, w, _ = image.shape
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    fig = plt.figure(figsize=(w*px, h*px))
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.axis('off')
    fig.add_axes(ax)
    '''
    need to specify axis range explicitly
    https://stackoverflow.com/questions/13018115/matplotlib-savefig-image-size-with-bbox-inches-tight
    '''
    im = ax.imshow(image, cmap=cmap, interpolation='nearest')
    
    '''
    Annotations
    https://matplotlib.org/1.5.3/users/annotations_guide.html   
    '''
    padding = 5
    offset_x, offset_y = -2, -2
    background = '#343837'   #'#363737'
    '''
    xkcd_fig = plot_colortable(mcolors.XKCD_COLORS)
    xkcd_fig.savefig("XKCD_Colors.png")
    https://xkcd.com/color/rgb/
    https://matplotlib.org/stable/gallery/color/named_colors.html
    '''

    boxprops = dict(fill = True, facecolor=background, edgecolor=background, pad=padding)

    """
    paddings
    https://stackoverflow.com/questions/38480739/aligning-a-text-box-edge-with-an-image-corner/38487750
    """
    if mytext is not None:
        an1 = ax.annotate(
            text = mytext,
            fontsize = 10,
            color='snow',
            xy=(0, 0),
            xytext=(padding-offset_x, -(padding-offset_y)),
            textcoords = 'offset pixels',
            bbox=boxprops,
            va='top',
            ha='left',
            clip_on = True,
            )

    return fig, ax, im

 

 