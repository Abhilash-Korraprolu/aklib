# V0.5

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

except ImportError:
    raise ImportError(
        "quickPlots require pandas, matplotlib, and seaborn to be installed") from None


def countplot(data1, column, x=6, y=6, palette="d", xlabel_name=None, ylabel_name=None,
              xlabel_size=16, ylabel_size=16, xticks_size=14, yticks_size=14, orientation='x',
              hue=None, save_name=None, folder_path='Plots/', dpi=500, legend_size=5,
              legend_loc='upper right', order=None, yrange_min=None, yrange_max=None, xrange_min=None,
              xrange_max=None):

    with plt.style.context('seaborn-whitegrid'):
        f, ax = plt.subplots(figsize=(x, y))

    '''
    Order Parameter:

    None (default) => default behavior
    'a'  => ascending
    'd'  => descending
    '[]' => pass a list for custom order.
    '''

    if order == 'a':
        order = list(data1[column].value_counts(ascending=True).index)
    elif order == 'd':
        order = list(data1[column].value_counts(ascending=False).index)

    if orientation.lower() == 'y':
        # When hue is present, we don't want to have palette colours.
        if hue == None:
            p1 = sns.countplot(y=column, data=data1, palette=(
                'Blues_'+palette), order=order)
        else:
            p1 = sns.countplot(y=column, data=data1, hue=hue, order=order)

        plt.xlabel(xlabel_name or 'Count', size=xlabel_size)
        plt.ylabel(ylabel_name or column.title(), size=ylabel_size)
    else:
        if hue == None:
            p1 = sns.countplot(x=column, data=data1, palette=(
                'Blues_'+palette), order=order)
        else:
            p1 = sns.countplot(x=column, data=data1, hue=hue, order=order)

        plt.xlabel(xlabel_name or column.title(), size=xlabel_size)
        plt.ylabel(ylabel_name or 'Count', size=ylabel_size)

    if hue != None:
        ax.legend(data1[column].unique(), loc=legend_loc,
                  prop={'size': legend_size})

    plt.xticks(size=xticks_size)
    plt.yticks(size=yticks_size)

    if xrange_min != None:
        plt.xlim(xrange_min, xrange_max)

    if yrange_min != None:
        plt.ylim(yrange_min, yrange_max)

    # UNDER CONSTRUCTION:
    # Adding % on the top of the bars. Doesn't work when orientation== 'y'.
    #total = len(data1[column])
    # for p in ax.patches:
        #percentage = '{:.1f}%'.format(100 * p.get_height()/total)

        #x = p.get_x() + p.get_width() / 2 - 0.05
        #y = p.get_y() + p.get_height()
        #ax.annotate(percentage, (x, y), size = 12)

    if save_name != None:
        f.savefig(folder_path + save_name + '.png',
                  dpi=dpi, bbox_inches='tight')

    return plt


def pie(data1, column, size=5, exp=None, legend=True, legend_loc='upper right', font_size=14, col_name=None,
        col_size=15, get_colors=False, colors_custom=None, save_name=None, folder_path='Plots/', dpi=500):

    # Issue: Fix the current limitation of n categories, here 10. The code fails when there are more than n categories.
    colors = ['#0fbcf9', '#ffc048', '#00d8d6', '#ef5777', '#05c46b',
              '#fa8231', '#fc5c65', '#fed330', '#26de81', '#45aaf2']
    # We can get the existing color codes to control which category gets which color.
    if get_colors == True:
        return colors

    with plt.style.context('default'):
        f, ax = plt.subplots(figsize=(size, size))

    default_exp = (0.02,)
    col_categories_count = data1[column].nunique()
    explode = exp or default_exp * col_categories_count  # to break the pie
    labels = data1[column].unique()

    data1[column].value_counts().plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': font_size},
                                      colors=colors_custom or colors, explode=explode)

    if legend == True:
        ax.legend(labels, loc=legend_loc)

    if col_name == None:
        plt.xlabel('')
        plt.ylabel('')
    else:
        plt.ylabel(col_name, size=col_size)

    if save_name != None:
        f.savefig(folder_path + save_name + '.png',
                  dpi=dpi, bbox_inches='tight')

    return plt


def hist(data1, column, bins=None, bin_size=1, edgecolor='black', median_axvline=False,
         median_name_axvline='Median', median_color_axvline='#fc4f30', mean_axvline=False,
         mean_name_axvline='Average', mean_color_axvline='#EAB543', x=10, y=6, save_name=None, legend_size=12,
         folder_path='Plots/', dpi=500, yrange_min=None, yrange_max=None, xrange_min=None, xrange_max=None,
         xlabel_name=None, ylabel_name=None, xlabel_size=16, ylabel_size=16):

    with plt.style.context('seaborn-whitegrid'):
        f, ax = plt.subplots(figsize=(x, y))

    col = data1[column]
    bins = bins or list(range(col.min(), col.max() + 1, bin_size))

    plt.hist(col, bins=bins, edgecolor=edgecolor)
    #sns.distplot(col, bins= bins, hist_kws=dict(edgecolor= edgecolor, linewidth=2));
    if median_axvline == True:
        plt.axvline(col.median(), color=median_color_axvline,
                    label=median_name_axvline)
        plt.legend()

    if mean_axvline == True:
        plt.axvline(col.mean(), color=mean_color_axvline,
                    label=mean_name_axvline)
        plt.legend()

    ax.legend(prop={'size': legend_size})

    if xrange_min != None:
        plt.xlim(xrange_min, xrange_max)

    if yrange_min != None:
        plt.ylim(yrange_min, yrange_max)

    plt.xlabel(xlabel_name or column.title(), size=xlabel_size)
    plt.ylabel(ylabel_name or 'Count', size=ylabel_size)

    if save_name != None:
        f.savefig(folder_path + save_name + '.png',
                  dpi=dpi, bbox_inches='tight')

    return plt


def heat(data1, x_col=None, y_col=None, type='%', x=6, y=6, xlabel_name=None, xlabel_size=14,
         ylabel_name=None, ylabel_size=14, xticks_size=12, yticks_size=12, save_name=None,
         dpi=500, folder_path='Plots/'):

    pivot = pd.pivot_table(data1, index=[y_col], columns=[
                           x_col], aggfunc='count', fill_value=0)
    temp = list(data1)
    temp.remove(x_col)
    temp.remove(y_col)
    pivot = pivot[temp[1]]

    if type == '%':
        tot_cols = len(data1)
        for col in data1[x_col].unique():
            pivot[col] = pivot[col].apply(lambda x: (x/tot_cols)*100)

    with plt.style.context('default'):
        f, ax = plt.subplots(figsize=(x, y))

    # Generate a custom diverging colormap
    cmap = sns.color_palette("Blues")

    # Draw the heatmap with the mask and correct aspect ratio
    if type == '%':
        ax = sns.heatmap(pivot, cmap=cmap, annot=True,
                         fmt='.1f', linewidth=2, square=True)
    else:
        ax = sns.heatmap(pivot, cmap=cmap, annot=True,
                         fmt='d', linewidth=2, square=True)

    plt.xlabel(xlabel_name or x_col, fontsize=xlabel_size)
    plt.ylabel(ylabel_name or y_col, fontsize=ylabel_size)
    plt.xticks(fontsize=xticks_size, va="center")
    plt.yticks(fontsize=yticks_size, va="center", rotation=0)

    if type == '%':
        for t in ax.texts:
            t.set_text(t.get_text() + " %")

    if save_name != None:
        f.savefig(folder_path + save_name + '.png',
                  dpi=dpi, bbox_inches='tight')

    return plt

# Boxplot & Violin plot
def bv(data1, cat_col=None, num_col=None, type='b', x=10, y=8, xlabel_name=None, xlabel_size=16,
       ylabel_name=None, ylabel_size=16, xticks_size=14, yticks_size=14, multicol_melt=False,
       save_name=None, dpi=500, folder_path='Plots/', yrange_min=0, yrange_max=None, xrange_min=0,
       xrange_max=None, xinterval=None, yinterval=None):

    with plt.style.context('seaborn-whitegrid'):
        f, ax = plt.subplots(figsize=(x, y))

    # For multicol_melt, x and y parameters are mandatory. They consist of label names
    if multicol_melt == True:
        data1 = pd.melt(data1)
        data1.columns = [cat_col, num_col]

    if type == 'b':
        p1 = sns.boxplot(x=cat_col, y=num_col, data=data1)
    else:
        p1 = sns.violinplot(x=cat_col, y=num_col, data=data1)

    if xrange_max != None:
        plt.xlim(xrange_min, xrange_max)

    if yrange_max != None:
        plt.ylim(yrange_min, yrange_max)

    # setting interval of ticks

    # ALTERT: No interval for cat col. It is only for events where x is also a num col.
    if xinterval != None:
        p1.set(xticks=[i for i in range(xrange_min, int(
            xrange_max or max(data[cat_col])) + 1, xinterval)])
    if yinterval != None:
        p1.set(yticks=[i for i in range(yrange_min, int(
            yrange_max or max(data[num_col])) + 1, yinterval)])

    return commons(plt, cat_col, num_col, xlabel_name, ylabel_name, xlabel_size, ylabel_size, xticks_size,
                   yticks_size, save_name, dpi, folder_path)


def rel(data1, x_col, y_col, xlabel_name=None, xlabel_size=16, ylabel_name=None, ylabel_size=16,
        xticks_size=14, yticks_size=14, save_name=None, dpi=500, folder_path='Plots/',
        height=6, aspect=1, ci=None, kind='line', hue=None, legend="brief", yrange_min=0,
        yrange_max=None, xrange_min=0, xrange_max=None, xinterval=None, yinterval=None):
    '''
    # Legend => "brief", "full", or False
    # Ci => either a percentage number such as 36.2, 95 etc. or 'sd' (Standard Deviation) or None
    # kind => 'scatter', 'line'
    '''

    p1 = None
    with plt.style.context('seaborn-whitegrid'):
        p1 = sns.relplot(x=x_col, y=y_col, data=data1, kind=kind,
                         ci=ci, hue=hue, height=height, aspect=aspect)

    # Setting range of ticks (without interval specification)
    if xrange_max != None and xinterval == None:
        plt.xlim(xrange_min, xrange_max)

    if yrange_max != None and yinterval == None:
        plt.ylim(yrange_min, yrange_max)

    # setting interval of ticks
    if xinterval != None:
        p1.set(xticks=[i for i in range(xrange_min, int(
            xrange_max or max(data[x_col])) + 1, xinterval)])
    if yinterval != None:
        p1.set(yticks=[i for i in range(yrange_min, int(
            yrange_max or max(data[y_col])) + 1, yinterval)])

    y_col = 'Average ' + y_col

    return commons(plt, x_col, y_col, xlabel_name, ylabel_name, xlabel_size, ylabel_size, xticks_size, yticks_size,
                   save_name, dpi, folder_path)


def bar(data1, cat_col, num_col, agg='mean', x=6, y=6, xlabel_name=None, xlabel_size=16,
        ylabel_name=None, ylabel_size=16, xticks_size=14, yticks_size=14, yrange_min=0, yrange_max=None,
        yinterval=None, save_name=None, dpi=500, folder_path='Plots/', order=None):

    temp = data1.groupby(by=cat_col)[num_col].agg([agg])
    temp = temp.reset_index()
    temp.columns = [cat_col, num_col]

    with plt.style.context('seaborn-whitegrid'):
        f, ax = plt.subplots(figsize=(x, y))

        if order == None:
            p1 = sns.barplot(x=cat_col, y=num_col, data=temp,
                             palette=('Blues_'+'d'))
        else:
            odr = False if order == 'd' else True
            temp.sort_values(by=num_col, ascending=odr, inplace=True)
            temp.reset_index(drop=True, inplace=True)
            p1 = sns.barplot(x=cat_col, y=num_col, data=temp, palette=(
                'Blues_'+'d'), order=temp[cat_col].values)

        temp[num_col].plot(zorder=2, color='red')

    import math
    if yrange_min != 0 or yrange_max != None:
        plt.ylim(yrange_min, yrange_max or math.ceil(
            max(temp[num_col]) + temp[num_col].std()))

    if yinterval != None:
        p1.set(yticks=[i for i in range(yrange_min, int(
            yrange_max or max(data[num_col])) + 1, yinterval)])

    # Note: Impacted by order
    for index, row in temp.iterrows():
        p1.text(row.name, row[num_col], round(
            row[num_col], 2), color='black', ha="center")

    if agg == 'mean':
        agg = 'average'
    num_col = agg+' '+num_col

    return commons(plt, cat_col, num_col, xlabel_name, ylabel_name, xlabel_size, ylabel_size, xticks_size,
                   yticks_size, save_name, dpi, folder_path)


def commons(plt, x_col, y_col, xlabel_name, ylabel_name, xlabel_size, ylabel_size, xticks_size, yticks_size,
            save_name, dpi, folder_path):

    plt.xlabel(xlabel_name or x_col.title(), fontsize=xlabel_size)
    plt.ylabel(ylabel_name or y_col.title(), fontsize=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size, rotation=0)

    if save_name != None:
        plt.savefig(folder_path + save_name + '.png',
                    dpi=dpi, bbox_inches='tight')

    return plt
