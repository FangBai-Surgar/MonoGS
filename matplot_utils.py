
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.text import OffsetFrom

# from mpl_toolkits.axes_grid.axes_grid import AxesGrid
# from mpl_toolkits.axes_grid.anchored_artists import AnchoredText





def image_annotation(image, cmap=plt.get_cmap('hot'), mytext = "my text"):
    """
    Annotate an image with text on the upper-left corner
    """

    fig, ax = plt.subplots(1)
    plt.imshow(image, cmap=cmap, interpolation='nearest')
    plt.axis('off')

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
    
    return fig, ax
 

 