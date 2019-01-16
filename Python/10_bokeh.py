# Basic plotting with Bokeh --------------------------------------------------------------------------------------------
import pandas as pd
dat = pd.read_csv('data/literacy_birth_rate.csv').dropna()
fertility = dat['fertility'].tolist()
fertility = [float(i) for i in fertility]
female_literacy = dat['female literacy'].tolist()
female_literacy = [float(i) for i in female_literacy]

# Scatter plots
from bokeh.plotting import figure
from bokeh.io import output_file, show

p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
p.circle(fertility, female_literacy, color='blue', size=10, alpha=0.8)
output_file('fert_lit.html')
show(p)

# Lines
stocks = pd.read_csv('data/stocks.csv', parse_dates=True, index_col='Date')
price = stocks['AAPL'].tolist()
date = stocks.index.tolist()

p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
p.line(date, price)
output_file('line.html')
show(p)

# Lines and markers
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
p.line(date[0:20], price[0:20])
p.circle(date[0:20], price[0:20], fill_color='white', size=4)
output_file('line.html')
show(p)

# Patches
# In Bokeh, extended geometrical shapes can be plotted by using the patches() glyph function. The patches glyph takes
# as input a list-of-lists collection of numeric values specifying the vertices in x and y directions of each distinct
# patch to plot.
# Plot the state borders of Arizona, Colorado, New Mexico and Utah. The latitude and longitude vertices for each state
# have been prepared as lists.
x = [az_lons, co_lons, nm_lons, ut_lons]
y = [az_lats, co_lats, nm_lats, ut_lats]
p.patches(x, y, line_color='white')
output_file('four_corners.html')
show(p)

# Plotting data from NumPy arrays
import numpy as np
x = np.linspace(0, 5, 100)
y = np.cos(x)
p = figure()
p.circle(x, y)
output_file('numpy.html')
show(p)

# Plotting data from Pandas DataFrames
df = pd.read_csv('data/auto.csv')
p = figure(x_axis_label='HP', y_axis_label='MPG')
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)
output_file('auto-df.html')
show(p)

# ColumnDataSource
# The ColumnDataSource is a table-like data object that maps string column names to sequences (columns) of data.
# It is the central and most common data structure in Bokeh.
from bokeh.plotting import ColumnDataSource
df = pd.read_csv('data/sprint.csv')

source = ColumnDataSource(df)
p = figure()
p.circle('Year', 'Time', source=source, color='color', size=8)
output_file('sprint.html')
show(p)

# Selection and non-selection glyphs
p = figure(x_axis_label='Year', y_axis_label='Time', tools='box_select')
p.circle('Year', 'Time', source=source, selection_color='red', nonselection_alpha = 0.1)
output_file('selection_glyph.html')
show(p)

# Hover glyphs
from bokeh.models import HoverTool
df = pd.read_csv('data/glucose.csv')

p = figure(x_axis_label='Time of day', y_axis_label='Blood glucose (mg/dL)')
p.circle(x=range(0, 288), y=df['glucose'], size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')
hover = HoverTool(tooltips=None, mode='vline')
p.add_tools(hover)
output_file('hover_glyph.html')
show(p)

# Colormapping
from bokeh.models import CategoricalColorMapper
df = pd.read_csv('data/auto.csv')

source = ColumnDataSource(df)
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])
p = figure()
p.circle('weight', 'mpg', source=source, color=dict(field='origin', transform=color_mapper), legend='origin')
output_file('colormap.html')
show(p)


# Layouts, Interactions, and Annotations -------------------------------------------------------------------------------
# Rows and columns of plots
from bokeh.layouts import row, column

layout = row(p1, p2)
output_file('row.html')
show(layout)

layout = column(p1, p2)
output_file('col.html')
show(layout)

row2 = column([p1, p2], sizing_mode='scale_width')
layout = row([p3, row2], sizing_mode='scale_width')
output_file('layout_custom.html')
show(layout)

# Gridplots
from bokeh.layouts import gridplot
# Give list of rows as input
# toolbar_location can be: 'above', 'below', 'left' or 'right'
layout = gridplot([[None, p1], [p2, p3]], toolbar_location=None)

# Tabbed layouts
from bokeh.model.widgets import Tabs, Panel
# Create a panel with a title for each tab
tab1 = Panel(child=row(p1, p2), title='first tab')
tab2 = Panel(child=p3, title='second tab')
# Put panels in a tabs object
layout = Tabs(tabs=[tab1, tab2])
output_file('tabbed.html')
show(layout)

# Linking plots together
# Linking axes
# (so that when one plot is zoomed or dragged, one or more of the other plots will respond)
# It's achieved by simply sharing range objects before using output_file(), show()
p2.x_range = p1.x_range
p2.y_range = p1.y_range

# Linked brushing
# By sharing the same ColumnDataSource object between multiple plots, selection tools like BoxSelect and LassoSelect
# will highlight points in both plots that share a row in the ColumnDataSource.

data = pd.read_csv('data/literacy_birth_rate.csv').dropna()
data['female literacy'] = pd.to_numeric(data['female literacy'])
source = ColumnDataSource(data)
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)',
            tools='box_select,lasso_select')
p1.circle('fertility', 'female literacy', source=source)
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)',
            tools='box_select,lasso_select')
p2.circle('fertility', 'population', source=source)
layout = row([p1, p2])
output_file('linked_brush.html')
show(layout)

# Annotations and legends
# Adding legend
import pandas as pd
from bokeh.plotting import ColumnDataSource, figure, output_file, show
data = pd.read_csv('data/literacy_birth_rate.csv').dropna()
data.rename(columns={'female literacy': 'female_literacy'}, inplace=True)
data['female_literacy'] = pd.to_numeric(data['female_literacy'])
latin_america = ColumnDataSource(data[data.Continent == 'LAT'])
africa = ColumnDataSource(data[data.Continent == 'AF'])
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)')
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')
output_file('fert_lit_groups.html')
show(p)

from bokeh.models import CategoricalColorMapper
df = pd.read_csv('data/auto.csv')
source = ColumnDataSource(df)
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])
p = figure()
p.circle('weight', 'mpg', source=source, color=dict(field='origin', transform=color_mapper), legend='origin')
output_file('colormap.html')
show(p)

# Positioning and styling legends
p.legend.location = 'bottom_left'
p.legend.background_fill_color = 'lightgray'
output_file('colormap.html')
show(p)

# Adding a hover tooltip
from bokeh.models import HoverTool
hover = HoverTool(tooltips=[('Model','@name')])
p.add_tools(hover)
output_file('hover.html')
show(p)


# Building interactive apps with Bokeh ---------------------------------------------------------------------------------
from bokeh.io import curdoc

# A plot
from bokeh.plotting import figure
plot = figure()
plot.line(x=[1,2,3,4,5], y=[2,5,4,6,7])
curdoc().add_root(plot)

# A slider
from bokeh.layouts import widgetbox
from bokeh.models import Slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)
layout = widgetbox(slider)
curdoc().add_root(layout)

# Connecting sliders to plots
from bokeh.plotting import ColumnDataSource, figure
from numpy.random import random
from bokeh.layouts import column
N = 300
source = ColumnDataSource(data={'x': random(N), 'y': random(N)})
# Create plots and widgets
plot = figure()
plot.circle('x', 'y', source=source)
slider = Slider(start=10, end=100, value=N, step=1, title='Number of points')
# Add callback to widgets
def callback(attr, old, new):
    N = slider.value
    source.data = {'x': random(N), 'y': random(N)}
slider.on_change('value', callback)
# Arrange stuff
layout = column(slider, plot)
# Display stuff
curdoc().add_root(layout)
# to show it, save in separate .py file and run
# bokeh serve --show myapp.py
# in the terminal

# Dropdown menus
menu = Select(options=['uniform', 'normal', 'lognormal'], value='uniform', title='Distribution')
def callback(attr, old, new):
    if menu.value == 'uniform': f = random
    elif menu.value == 'normal': f = normal
    else: f = lognormal
    source.data = {'x': f(size=N), 'y': f(size=N)}
menu.on_change('value', callback)

# Another example of dropdown menu
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

plot = figure()
plot.circle('x', 'y', source=source)

def update_plot(attr, old, new):
    if new == 'female_literacy':
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }


select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')
select.on_change('value', update_plot)
layout = row(select, plot)
curdoc().add_root(layout)

# Synchronize two dropdowns
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')

def callback(attr, old, new):
    # If select1 is 'A'
    if select1.value == 'A':
        select2.options = ['1', '2', '3']
        select2.value = '1'
    else:
        select2.options = ['100', '200', '300']
        select2.value = '100'


select1.on_change('value', callback)
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

# Buttons
from bokeh.model import Button
button = Button(label='press me')
def update():
    # do something

button.on_click(update)

# Button types
from bokeh.model import CheckboxGroup, RadioGroup, Toggle
toggle = Toggle(label='some on/off', button_type='success')
checkbox = CheckboxGroup(labels=['foo', 'bar', 'baz'])      # many can be chosen
radio = RadioGroup(labels=['2000', '2010', '2020'])         # only one can be chosen
def callback(active):
    # Active tells which button is active

curdoc().add_root(widgetbox(toggle, checkbox, radio))

