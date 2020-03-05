# Import the necessary modules
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6
from bokeh.layouts import widgetbox, row
from bokeh.models import Slider, Select
from bokeh.io import curdoc
import pandas as pd

# Get data and source it
data = pd.read_csv('data/gapminder_tidy.csv', index_col='Year')
source = ColumnDataSource(data={
    'x': data.loc[1970].fertility,
    'y': data.loc[1970].life,
    'country': data.loc[1970].Country,
    'pop': (data.loc[1970].population / 20000000) + 2,
    'region': data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)
# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Prepare color mapper
regions_list = data.region.unique().tolist()
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Create the plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

plot.circle(x='x', y='y', fill_alpha=0.8, source=source, legend='region',
            color=dict(field='region', transform=color_mapper))

plot.legend.location = 'top_right'
plot.xaxis.axis_label ='Fertility (children per woman)'
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Createa hover tool and add it to the plot
hover = HoverTool(tooltips=[('Country', '@country')])
plot.add_tools(hover)

# Define the callback function to update the plot based on slider
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x': data.loc[yr].fertility,
        'y': data.loc[yr].life,
        'country': data.loc[yr].Country,
        'pop': (data.loc[yr].population / 20000000) + 2,
        'region': data.loc[yr].region,
    }
    # Update source data
    source.data = new_data
    # Update plot title
    plot.title.text = 'Gapminder data for %d' % yr
    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])


# Make a slider object and attach the callback to its 'value' property
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x and y data and attach the callback to their 'value' properties
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)
x_select.on_change('value', update_plot)
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)
curdoc().title = 'Gapminder'

# to run, in terminal type: bokeh serve --show bokeh_app_example.py