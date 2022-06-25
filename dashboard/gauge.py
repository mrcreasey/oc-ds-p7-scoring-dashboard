import os, sys
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge, Rectangle

def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points


def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation


def plot_gauge(arrow=0.5, labels=['LOW','MEDIUM','HIGH','EXTREME'], 
          title='',min_val=0.,max_val=1.,threshold=-1.0,
          colors='RdYlGn', n_colors=-1, ax=None,figsize=(3,2)): 
    
    N=len(labels)
    n_colors = n_colors if n_colors > 0 else N
    if isinstance(colors, str):
        #  matplotlib colormap
        cmap = cm.get_cmap(colors, n_colors)
        cmap = cmap(np.arange(n_colors))
        # colors = cmap[::-1,:].tolist()
        colors = cmap
    if isinstance(colors, list): 
        n_colors= len(colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ang_range, _ = degree_range(n_colors)
    
    # plots the sectors and the arcs
    for ang, c in zip(ang_range, colors): 
        ax.add_patch(Wedge((0.,0.), .4, *ang, width=0.10, facecolor='w', lw=2, alpha=0.5))
        ax.add_patch(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=0, alpha=0.8))

    
    # labels (e.g. 'LOW','MEDIUM',...)
    _, mid_points = degree_range(N)
    labels = labels[::-1]
    a=0.45
    for mid, lab in zip(mid_points, labels): 
        ax.text(a * np.cos(np.radians(mid)), a * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=12, \
            fontweight='bold', rotation = rot_text(mid))


    # bottom banner and title
    ax.add_patch(Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2))    
    ax.text(0, -0.10, title, horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight='bold')
    
    # threshold
    if threshold>min_val and threshold<max_val:
        pos=180 * (max_val-threshold)/(max_val-min_val)
        a=0.25; b=0.18; x=np.cos(np.radians(pos)); y= np.sin(np.radians(pos))
        ax.arrow(a*x, a*y,b*x,b*y, width=0.01, head_width=0.0, head_length=0, ls='--',fc='r', ec='r')

    # arrow
    pos=180 * (max_val-arrow)/(max_val-min_val)
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    #ax.add_patch(Circle((0, 0), radius=0.01, facecolor='b', zorder=11))
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    return ax
