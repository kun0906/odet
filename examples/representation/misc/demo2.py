"""
'break axis'
https://stackoverflow.com/questions/18551633/how-to-set-an-axis-interval-range-using-matplotlib-or-other-libraries-in-python

https://matplotlib.org/examples/pylab_examples/broken_axis.html
https://stackoverflow.com/questions/53642861/broken-axis-slash-marks-inside-bar-chart-in-matplotlib


'zoom part of plot'
https://stackoverflow.com/questions/26223937/matplotlib-pandas-zoom-part-of-a-plot-with-time-series
https://pythonhosted.org/plottools/generated/plottools.zoom_axes.html
https://stackoverflow.com/questions/34952752/how-to-zoom-a-part-of-plot-by-matplolib
https://github.com/JuliaPlots/Plots.jl/issues/315


"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

flg = 0
if flg:
    penguins = sns.load_dataset("penguins")
    print(penguins)
    sns.histplot(data=penguins, x="flipper_length_mm", hue="species")
else:
    X = [1, 20, 3, 4, 5, 1, 2, 3, 90, 5, 6]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    y = ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
    s = min(len(X), len(y))
    data = np.array(list(zip(X[:s], y[:s])), dtype='O')
    df = pd.DataFrame(data, columns=[f'x', 'y'])
    sns.histplot(data=df, x='x', hue='y')
plt.show()
