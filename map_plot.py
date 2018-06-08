import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon


def main(
    states_to_values_dict={},
    title="",
    figname="US map plot",
    figsize=(),
    cal_blue_tex_red=True,
    cmap=plt.cm.jet,
):
    if figsize:
        plt.figure(figname, figsize=figsize)
    else:
        plt.figure(figname)

    if not states_to_values_dict:
        states_to_values_dict = example_dict()

    if cal_blue_tex_red:
        dictionary = dict_to_cal_blue_tex_red(states_to_values_dict)
    else:
        dictionary = states_to_values_dict

    plot_map(dictionary, title, cmap)


def plot_map(states_to_values_dict={},
             title="",
             cmap=plt.cm.jet):
    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    # draw state boundaries.
    # data from U.S Census Bureau
    # http://www.census.gov/geo/www/cob/st2000.html
    shp_info = m.readshapefile('st99_d00','states',drawbounds=True)

    # choose a color for each state based on population density.
    colors={}
    statenames=[]
    # cmap = plt.cm.jet # use 'hot' colormap
    vals = [value for key, value in states_to_values_dict.iteritems()]
    vmin = float(min(vals)); vmax = float(max(vals)) # set range.

    for shapedict in m.states_info:
        statename = shapedict['NAME']

        # skip DC and Puerto Rico.
        if statename not in ['Hawaii','Alaska', 'District of Columbia','Puerto Rico']:
            v = states_to_values_dict[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            colors[statename] = cmap((v-vmin)/(vmax-vmin))[:3]

        statenames.append(statename)

    # cycle through state names, color each one.
    ax = plt.gca() # get current axes instance

    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Hawaii','Alaska','District of Columbia','Puerto Rico']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)

    plt.title(title)
    plt.show()


def example_dict():
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
        'Colorado', 'Connecticut', 'Delaware', 'Dist. of Col.', 'Florida',
        'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'CD-1', 'CD-2', 'CD-3',
        'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
        'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
        'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'West Virginia', 'Wisconsin', 'Wyoming',
    ]
    return dict(zip(states, np.arange(len(states))))


def dict_to_cal_blue_tex_red(dictionary={}):
    vals = [value for key, value in dictionary.iteritems()]
    unique_vals = np.unique(vals)
    new_vals = np.arange(len(unique_vals))

    old_cal = dictionary["California"]
    old_tex = dictionary["Texas"]

    transform_dict = {old_cal: new_vals[0], old_tex: new_vals[-1]}
    unique_vals = list(unique_vals)
    unique_vals.remove(old_cal)
    unique_vals.remove(old_tex)
    new_vals = new_vals[1:-1]

    for old_val, new_val in zip(unique_vals, new_vals):
        transform_dict[old_val] = new_val

    new_dictionary = {}

    for key, value in dictionary.iteritems():
        new_dictionary[key] = transform_dict[value]

    return new_dictionary