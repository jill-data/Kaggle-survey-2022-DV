# %%
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy
from matplotlib import pyplot, transforms

# %%
df = pd.read_csv('./data/kaggle_survey_2022_responses.csv', skiprows=[0])

# %%
col_mapping = pd.read_csv('./data/column_mapping.csv')

# %%
col_mapping_dict = {k:v for k, v in col_mapping.dropna().values}

# %%
df = df[col_mapping_dict.keys()].rename(columns=col_mapping_dict)

# %%
df

# %%
df = df.query('student == "No"')

# %%
#Drop the column student after filtering to focus on industry insights from professionals
df = df.drop(columns=['student'])
df

# %%
df['title'].unique()

# %%
df = df.query('title.notna() and title != "Currently not employed"')


# %%
df['title'].unique()

# %%
df

# %%
df = df.query('industry.notna()')
df

# %%
#Plotting the proportions of country 
num_country = df['country'].value_counts()
num_country = num_country.reset_index()
num_country


# %% [markdown]
# ## 1. Popular positions from surveyors and their distribution by industry

# %%
position = df['title'].value_counts()
position = position.reset_index()
position

# %%
position['Group'] = position['index'].where(position['title'] > 100, 'Other')
position

# %%
group_position = position.groupby('Group')['title'].sum().reset_index().sort_values(by='title', ascending=False)
group_position



# %%
def plot_positions(group_position, ax):
    labels = group_position['Group']
    sizes = group_position['title']
    colors = ['#A6ABAD','#00587A', '#0073A1', '#00A1E0','#00BCE3','#87CEEB', '#89BCC4', '#9BD3DD', '#A4E0EB']

    
    patches, labels_, percentages = ax.pie(
        sizes, colors=colors,
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
        textprops={'color': 'white', 'fontweight': 'bold','fontname': "Sans Serif"},
        startangle=90, frame=True,
        autopct="%.2f%%",
        pctdistance=0.85,
    )

    ax.axis('off')

    ax.add_artist(plt.Circle((0, 0), 0.6, color='white', linewidth=0))

    # Adding Title of chart
    ax.set_title('Popular positions from surveyors', fontweight = 'bold', size = 15, fontfamily='serif', ha="center", color="#4d4d4d")

    ax.legend(labels, loc='upper right', bbox_to_anchor=(1.35, 0.75))


# %%
fig, axs = plt.subplots(figsize=(15,9), dpi=400)
plot_positions(group_position, axs)
plt.show()

# %% [markdown]
# ## Positions distribution in industry

# %%
df['count'] = 1

# %%
position_order = df['title']

industry_order = df['industry']


data_q5q15 = pd.pivot_table(df, values='count', index=['title'], columns=['industry'], aggfunc=np.sum).fillna(0).astype(int).loc[position_order, industry_order].stack()
data_q5q15_man = pd.pivot_table(df[df['gender']=='Man'], values='count', index=['title'], columns=['industry'], aggfunc=np.sum).fillna(0).astype(int).loc[position_order, industry_order].stack()
data_q5q15_woman = pd.pivot_table(df[df['gender']=='Woman'], values='count', index=['title'], columns=['industry'], aggfunc=np.sum).fillna(0).astype(int).loc[position_order, industry_order].stack()

# %%
def drawPieMarker(xs, ys, ratios, sizes, colors, ax):
    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x  = np.array([0] + np.cos(np.linspace(previous, this, 30)).tolist() + [0])
        y  = np.array([0] + np.sin(np.linspace(previous, this, 30)).tolist() + [0])
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes), 'facecolor':color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker, alpha=0.7)

# %%
fig = plt.figure(figsize=(20, 23), dpi=200)

gs = fig.add_gridspec(5, 5)



ax_plot = fig.add_subplot(gs[1:4, 0:4]) 
for q5_idx in position_order[::-1]:
    for q15_idx in industry_order:
        man = data_q5q15_man[q5_idx][q15_idx]
        woman = data_q5q15_woman[q5_idx][q15_idx]
        tot = data_q5q15[q5_idx][q15_idx]
        drawPieMarker([q15_idx],[q5_idx], [man/(man+woman), woman/(man+woman)] ,[tot*2.5], ['#004c70', '#990000'], ax=ax_plot)

ax_plot.grid(linewidth=0.2, zorder=0)        

ax_plot.set_yticklabels(q5_idx, fontfamily='serif', fontsize=15)
ax_plot.set_xticklabels(q15_idx, fontfamily='serif', fontsize=15, rotation=90)

# Pos
ax_pos = fig.add_subplot(gs[0, :4], sharex=ax_plot) 
data_q15_woman = df[df['gender']=='Woman']['industry'].value_counts()[industry_order]
ax_pos.bar(data_q15_woman.index, data_q15_woman, width=0.45, alpha=0.7, color='#990000')

data_q15_man = df[df['gender']=='Man']['industry'].value_counts()[industry_order]
ax_pos.bar(data_q15_man.index, data_q15_man, bottom=data_q15_woman , width=0.45, alpha=0.7, color='#004c70')

plt.setp(ax_pos.get_xticklabels(), visible=False)


# Exp
ax_exp = fig.add_subplot(gs[1:4, 4], sharey=ax_plot) 

data_q5_woman = df[df['gender']=='Woman']['title'].value_counts()[industry_order]
ax_exp.barh(data_q5_woman.index[::-1], data_q5_woman[::-1], height=0.55, alpha=0.7, color='#990000')

data_q5_man = df[df['gender']=='Man']['title'].value_counts()[industry_order]
ax_exp.barh(data_q5_man.index[::-1], data_q5_man[::-1], left= data_q5_woman[::-1],height=0.55, alpha=0.7, color='#004c70')

plt.setp(ax_exp.get_yticklabels(), visible=False)

# Spines
for s in ['top', 'left', 'right', 'bottom']:
    ax_plot.spines[s].set_visible(False)
    ax_pos.spines[s].set_visible(False)
    ax_exp.spines[s].set_visible(False)
    

fig.text(0.8, 0.9, 'Gender & Position & ML Experience', fontweight='bold', fontfamily='serif', fontsize=35, ha='right') 
fig.text(0.8, 0.88, 'Stacked Bar Chart + Categorical Bubble Pie Chart', fontweight='light', fontfamily='serif', fontsize=20, ha='right')
# plt.tight_layout()
plt.show()

# %%



