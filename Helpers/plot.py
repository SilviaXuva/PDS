import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from Helpers.log import Paths
from Helpers.input import Motion

def SaveData(file_name, velocity, real, kalman, integration, camera=None):
    df = pd.DataFrame(data={"velocity": velocity}, index=np.array([i*Motion.ts for i in range(len(velocity))]))
    df.to_csv(fr'{Paths.execution}\velocity.csv', sep=',')

    if camera is None:
        df = pd.DataFrame(data={"real": real, "kalman": kalman, "integration": integration}, index=np.array([i*Motion.ts for i in range(len(velocity))]))
    else:
        df = pd.DataFrame(data={"real": real, "kalman": kalman, "integration": integration, "camera": camera}, index=np.array([i*Motion.ts for i in range(len(velocity))]))
    df.to_csv(fr'{Paths.execution}\{file_name}.csv', sep=',')

def Plot(file_name, camera=False):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    df_vel = pd.read_csv(fr'{Paths.execution}\velocity.csv', index_col=0)
    df = pd.read_csv(fr'{Paths.execution}\{file_name}.csv', index_col=0)

    if camera:
        styles = ['b-','r--','g-.','m-']
        linewidths = [1, 1, 1, 1]
    else:
        styles = ['b-','r--','g-.']
        linewidths = [1, 1, 1]
    df_vel.plot(ax=axes[0])

    for column, style, lw in zip(df.columns, styles, linewidths):
        df[column].plot(ax=axes[1], style=style, linewidth=lw)
    
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()  # Maximizes the figure window

    axes[1].legend(df.columns)  # Adding legend based on DataFrame column names
    
    plt.show()
    fig.savefig(fr'{Paths.execution}\{file_name}.png')

def Plot_(executionPath, file_name, camera=False):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    df_vel = pd.read_csv(fr'{executionPath}\velocity.csv', index_col=0)
    df = pd.read_csv(fr'{executionPath}\{file_name}.csv', index_col=0)

    if camera:
        styles = ['b-','r--','g-.','m-']
        linewidths = [1, 1, 1, 1]
    else:
        styles = ['b-','r--','g-.']
        linewidths = [1, 1, 1]
    df_vel.plot(ax=axes[0])

    for column, style, lw in zip(df.columns, styles, linewidths):
        df[column].plot(ax=axes[1], style=style, linewidth=lw)
    
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()  # Maximizes the figure window

    axes[1].legend(df.columns)  # Adding legend based on DataFrame column names
    
    plt.show()
    fig.savefig(fr'{executionPath}\{file_name}.png')