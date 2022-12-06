import warnings

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.text import OffsetFrom
from pandas.plotting import parallel_coordinates

df = pd.read_csv('./data/kaggle_survey_2022_responses.csv', skiprows=[0])

# # Data cleaning

# ## Columns

col_mapping = pd.read_csv('./data/column_mapping.csv')

col_mapping_dict = {k: v for k, v in col_mapping.dropna().values}

df = df[col_mapping_dict.keys()].rename(columns=col_mapping_dict)

# ## Select data professionalas only
#

pos_mapping = pd.read_csv('./data/position_mapping.csv')
pos_maping_dict = {k: v for k, v in pos_mapping.dropna().values}

df['title'] = df['title'].map(pos_maping_dict)

df = df.query('student == "No"')

# Drop the column student after filtering to focus on industry insights from professionals
df = df.drop(columns=['student'])
df.head(3)

df = df.query('title.notna() and title != "Currently not employed"')


df['title'].unique()

df = df.query('industry.notna()')
df.head(3)

# Plotting the proportions of country
num_country = df['country'].value_counts()
num_country = num_country.reset_index()
num_country

# Encode Machine learning stages

df['company_stage'].unique()

company_stage_dict = {
    'I do not know': -1,
    'No (we do not use ML methods)': 0,
    'We are exploring ML methods (and may one day put a model into production)': 1,
    'We use ML methods for generating insights (but do not put working models into production)': 2,
    'We recently started using ML methods (i.e., models in production for less than 2 years)': 3,
    'We have well established ML methods (i.e., models in production for more than 2 years)': 4
}


df['stage_index'] = df['company_stage'].map(company_stage_dict)

# ML experience

df['ml_seniority'] = df['ml_exp'].map({
    'I do not use machine learning methods': 0,
    'Under 1 year': 0.5,
    '1-2 years': 1.5,
    '2-3 years': 2.5,
    '3-4 years': 3.5,
    '4-5 years': 4.5,
    '5-10 years': 7.5,
    '10-20 years': 15,
})

df['salary_usd'] = df['yearly_compensation'].map({
    '25,000-29,999': 27499.5,
    '100,000-124,999': 112499.5,
    '200,000-249,999': 224999.5,
    '150,000-199,999': 174999.5,
    '90,000-99,999': 94999.5,
    '30,000-39,999': 34999.5,
    '3,000-3,999': 3499.5,
    '50,000-59,999': 54999.5,
    '125,000-149,999': 137499.5,
    '15,000-19,999': 17499.5,
    '5,000-7,499': 6249.5,
    '10,000-14,999': 12499.5,
    '20,000-24,999': 22499.5,
    '$0-999': 499.5,
    '7,500-9,999': 8749.5,
    '4,000-4,999': 4499.5,
    '80,000-89,999': 84999.5,
    '2,000-2,999': 2499.5,
    '250,000-299,999': 274999.5,
    '1,000-1,999': 1499.5,
    '$500,000-999,999': 749999.5,
    '70,000-79,999': 74999.5,
    '60,000-69,999': 64999.5,
    '40,000-49,999': 44999.5,
    '300,000-499,999': 399999.5,
    '>$1,000,000': 1000000,
})

df['coding_exp_year'] = df['code_exp'].map({
    '10-20 years': 15,
    '20+ years': 20,
    '1-3 years': 2,
    '5-10 years': 7.5,
    '3-5 years': 4,
    '< 1 years': 0.5,
    'I have never written code': 0,
})

df.to_csv('./data/cleaned_kaggle2022.csv', index=False)

# ## 1. Gender & Title distribution in different industries

position = df['title'].value_counts()
position = position.reset_index()

position['Group'] = position['index'].where(position['title'] > 100, 'Other')

group_position = position.groupby('Group')['title'].sum().reset_index().sort_values(by='title', ascending=False)


def plot_positions(group_position, ax):
    labels = group_position['Group']
    sizes = group_position['title']
    colors = ['#A6ABAD', '#00587A', '#0073A1', '#00A1E0', '#00BCE3', '#87CEEB', '#89BCC4', '#9BD3DD', '#A4E0EB']

    patches, labels_, percentages = ax.pie(
        sizes, colors=colors,
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
        textprops={'color': 'white', 'fontweight': 'bold', 'fontname': "Sans Serif"},
        startangle=90, frame=True,
        autopct="%.2f%%",
        pctdistance=0.85,
    )

    ax.axis('off')

    ax.add_artist(plt.Circle((0, 0), 0.6, color='white', linewidth=0))

    # Adding Title of chart
    ax.set_title('Popular positions from surveyors', fontweight='bold', size=15, fontfamily='serif', ha="center", color="#4d4d4d")

    ax.legend(labels, loc='upper right', bbox_to_anchor=(1.35, 0.75))


# fig, axs = plt.subplots(figsize=(15,9), dpi=400)
# plot_positions(group_position, axs)
# plt.show()

df['count'] = 1

position_order = sorted(df['title'].unique().tolist())
position_order

industry_order = sorted(df['industry'].unique().tolist())
industry_order

industry_title_df = pd.pivot_table(
    df,
    values='count',
    index=['title'],
    columns=['industry'],
    aggfunc=np.sum,
).fillna(0).astype(int).loc[position_order, industry_order].stack()

industry_title_df_man = pd.pivot_table(
    df[df['gender'] == 'Man'],
    values='count',
    index=['title'],
    columns=['industry'],
    aggfunc=np.sum,
).fillna(0).astype(int).loc[position_order, industry_order].stack()

industry_title_df_woman = pd.pivot_table(
    df[df['gender'] == 'Woman'],
    values='count',
    index=['title'],
    columns=['industry'],
    aggfunc=np.sum,
).fillna(0).astype(int).loc[position_order, industry_order].stack()


def drawPieMarker(xs, ys, ratios, sizes, colors, ax):
    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x = [0] + np.cos(np.linspace(previous, this, 30)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 30)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker': xy, 's': np.abs(xy).max()**2 * np.array(sizes), 'facecolor': color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker, alpha=0.7)


with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    fig = plt.figure(figsize=(22, 25), dpi=200)

    gs = fig.add_gridspec(5, 5)

    ax_plot = fig.add_subplot(gs[1:4, 0:4])
    for q5_idx in position_order[::-1]:
        for q15_idx in industry_order:
            man = industry_title_df_man[q5_idx][q15_idx]
            woman = industry_title_df_woman[q5_idx][q15_idx]
            tot = industry_title_df[q5_idx][q15_idx]
            drawPieMarker(
                [q15_idx],
                [q5_idx],
                [man / (man + woman), woman / (man + woman)],
                [tot * 7],
                ['#1170aa', '#B21807'],
                ax=ax_plot,
            )

    ax_plot.grid(linewidth=0.2, zorder=0)

    ax_plot.tick_params(axis='x', labelrotation=90)

    # region Industry
    ax_int = fig.add_subplot(gs[0, :4], sharex=ax_plot)
    data_industry_woman = df[df['gender'] == 'Woman']['industry'].value_counts()[industry_order]
    ax_int.bar(
        data_industry_woman.index,
        data_industry_woman,
        width=0.45,
        alpha=0.7,
        color='#B21807',
        label='Female',
    )

    data_industry_man = df[df['gender'] == 'Man']['industry'].value_counts()[industry_order]
    ax_int.bar(
        data_industry_man.index,
        data_industry_man,
        bottom=data_industry_woman,
        width=0.45,
        alpha=0.7,
        color='#1170aa',
        label='Male',
    )

    plt.setp(ax_int.get_xticklabels(), visible=False)
    # endregion

    # region Title
    ax_tit = fig.add_subplot(gs[1:4, 4], sharey=ax_plot)

    data_q5_woman = df[df['gender'] == 'Woman']['title'].value_counts()[position_order]
    ax_tit.barh(data_q5_woman.index[::-1], data_q5_woman[::-1], height=0.55, alpha=0.7, color='#B21807')

    data_q5_man = df[df['gender'] == 'Man']['title'].value_counts()[position_order]
    ax_tit.barh(data_q5_man.index[::-1], data_q5_man[::-1], left=data_q5_woman[::-1], height=0.55, alpha=0.7, color='#1170aa')

    plt.setp(ax_tit.get_yticklabels(), visible=False)
    # endregion

    # Spines
    for s in ['top', 'left', 'right', 'bottom']:
        ax_plot.spines[s].set_visible(False)
        ax_int.spines[s].set_visible(False)
        ax_tit.spines[s].set_visible(False)

    fig.text(
        0.6,
        0.9,
        'Gender & Title distribution by Industry',
        fontweight='bold',
        fontfamily='arial',
        fontsize=35,
        ha='right',
        color='#C41E3A',
    )
    fig.text(
        0.6,
        0.88,
        'Source: Data Professionals - Kaggle Survey 2022',
        fontweight='light',
        style='italic',
        fontfamily='arial',
        fontsize=15,
        ha='right',
    )

    # Legend
    legend_gender = ax_int.legend(
        bbox_to_anchor=(1.2, 1.1),
        fontsize=16,
        frameon=False,
        title='Gender',
        title_fontsize=20,
    )
    legend_count = ax_int.annotate(
        'Count of professionals',
        xy=(0.1, 0.1),
        textcoords=OffsetFrom(legend_gender, (0.5, -0.5)),
        xytext=(0, 0),
        fontsize=20,
        ha='center',
    )

    corner_ax = fig.add_subplot(gs[0, 4], zorder=-1)
    corner_ax.set_axis_off()
    for s in ['top', 'left', 'right', 'bottom']:
        corner_ax.spines[s].set_visible(False)

    ax_count = corner_ax.inset_axes([0, 0, 1, 0.5])
    ax_count.set_axis_off()
    sizes = [50, 100, 200]
    x_coords = [0] * len(sizes)
    y_coords = np.arange(len(sizes))
    ax_count.scatter(x_coords, y_coords, marker='o', s=[size * 7 for size in sizes], c='#a3acb9', alpha=0.7)
    ax_count.set_xlim(-0.02, 0.06)
    ax_count.set_ylim(-0.5, 2.5)

    for size, x, y in zip(sizes, x_coords, y_coords):
        ax_count.annotate(str(size), (x + 0.015, y - 0.15), fontsize=16)

    plt.savefig('./charts/gender title by industry.png')
    plt.close('all')

# ## 2. ML in research by data professionals

df = df.query('education_level.notna() and education_level != "I prefer not to answer"')
df['education_level'].unique()

df['edu_group'] = df['education_level'].where(
    ~df['education_level'].isin([
        "No formal education past high school",
        "Some college/university study without earning a bachelor’s degree"
    ]),
    "Below Bachelor's Degree"
)

df['edu_group'] = df['edu_group'].where(
    ~df['edu_group'].isin([
        "Professional doctorate",
        "Doctoral degree"
    ]),
    "Above Master's Degree"
)

df['edu_group'].unique()

df_theo = df.query('ml_used_theoretical.notna()').groupby(by='title').count()
df_theo


ml_used_df = df.dropna(subset=['published'], how='all')
title_count = ml_used_df.groupby('title').size()
ml_theoretical = ml_used_df.groupby('title')['ml_used_theoretical'].count()
ml_applied = ml_used_df.groupby('title')['ml_used_applied'].count()
ml_no = ml_used_df.groupby('title')['ml_used_no'].count()
ml_used = pd.concat([title_count, ml_theoretical, ml_applied, ml_no], axis=1)
ml_used = ml_used.rename(columns={0: 'count'}).reset_index()
ml_used = (
    ml_used
    .eval('pct_theoretical = ml_used_theoretical / count * 100')
    .eval('pct_applied = ml_used_applied / count * 100')
    .eval('pct_no= ml_used_no / count * 100')
    .drop(['ml_used_theoretical', 'ml_used_applied', 'ml_used_no', 'count'], axis=1)
    .sort_values('pct_theoretical')
    .reset_index(drop=True)
)
ml_used

# Plot
plt.figure(figsize=(10, 7), dpi=200)
parallel_coordinates(ml_used, 'title', colormap='tab20')
labels = ['% Theoretical Research', '% Applied Research', '% No Research']

# Lighten borders
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)

plt.title(
    'Use of Machine Learning in published Research by Title\n',
    fontsize=15,
    fontweight='bold',
    ha='center',
    fontname='arial',
    color='#C41E3A',
)
plt.grid(alpha=0.3)
plt.xticks([0, 1, 2], labels, fontsize=9, fontweight='bold')
plt.yticks(fontsize=9, fontweight='bold')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('./charts/Use of ML.png')
plt.close('all')


value_range = [0, 55]

dimensions = [
    dict(
        range=[0, ml_used.index.max()],
        label='<b>Title</b>',
        values=ml_used.index,
        tickvals=ml_used.index,
        ticktext=ml_used['title'],
    ),
    dict(
        range=value_range,
        label='<b>% Theoretical Research</b>',
        values=ml_used['pct_theoretical']
    ),
    dict(
        range=value_range,
        label='<b>% Applied Research</b>',
        values=ml_used['pct_applied']
    ),
    dict(
        range=value_range,
        label='<b>% Not used</b>',
        values=ml_used['pct_no']
    ),
]

fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=ml_used.index,
            colorscale=px.colors.qualitative.G10,
        ),
        dimensions=dimensions,
    ),
)

fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    margin=dict(
        l=150,
        b=50,
        t=100,
    ),
    title='<b>Use of Machine Learning in published research by title</b>',
    title_font_size=20,
    title_x=0.5,
    title_font_family='arial',
    title_font_color='#C41E3A',
)

fig.show()

fig.write_image('./charts/Use of ML in research_PL.png', scale=2)

# ## 3.Essential tasks and skill sets per Role

# Load packages

df.head()

title_count = df.groupby('title').size()
title_count

python = df.groupby('title')['program_lang_Python'].count()
python

tasks = [
    'task_ Analyze and understand data to influence product or business decisions',
    'task_ Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
    'task_ Build prototypes to explore applying machine learning to new areas',
    'task_ Build and/or run a machine learning service that operationally improves my product or workflows',
    'task_ Experimentation and iteration to improve existing ML models',
    'task_ Do research that advances the state of the art of machine learning',
    'task_ None of these activities are an important part of my role at work',
]

languages = [
    'program_lang_Python',
    'program_lang_R',
    'program_lang_SQL',
    'program_lang_C',
    'program_lang_C#',
    'program_lang_C++',
    'program_lang_Java',
    'program_lang_Javascript',
    'program_lang_Bash',
    'program_lang_PHP',
    'program_lang_MATLAB',
    'program_lang_Julia',
    'program_lang_Go',
]

titles = df['title'].unique().tolist()
titles.remove('Other')

language_task = []
for task in tasks:
    for language in languages:
        language_task.append([
            language.replace('program_lang_', ''),
            task.split('_')[1][1:],
            df[df[task].notna() & df[language].notna()].shape[0],
        ])

title_language = []
for title in titles:
    for language in languages:
        title_language.append([
            title,
            language.replace('program_lang_', ''),
            df.query('title == @title').dropna(subset=[language]).shape[0] / df.query('title == @title').shape[0],
        ])

title_task = []
for title in titles:
    for task in tasks:
        title_task.append([
            title,
            task.split('_')[1][1:],
            df.query('title == @title').dropna(subset=[task]).shape[0] / df.query('title == @title').shape[0],
        ])

language_task_df = pd.DataFrame(language_task, columns=['source', 'target', 'value'])
title_language_df = pd.DataFrame(title_language, columns=['source', 'target', 'value'])
title_task_df = pd.DataFrame(title_task, columns=['source', 'target', 'value'])

sankey_df = pd.concat([language_task_df, title_language_df], ignore_index=True)
sankey_df

sankey_df

# Text to numeric for sankey plot
labels = list(set(sankey_df['source'].unique().tolist() + sankey_df['target'].unique().tolist()))
label_code = {label: code for code, label in enumerate(labels)}

sankey_df['source_code'] = sankey_df['source'].map(label_code)
sankey_df['target_code'] = sankey_df['target'].map(label_code)
sankey_df

# import plotly.graph_objects as go

# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#       pad=15,
#       thickness=20,
#       line=dict(color="black", width=0.5),
#       label=list(labels),
#       color="blue"
#     ),
#     link=dict(
#       source=sankey_df['source_code'].tolist(), # indices correspond to labels, eg A1, A2, A1, B1, ...
#       target=sankey_df['target_code'].tolist(),
#       value=sankey_df['value'].tolist(),
#   ))])

# fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
# fig.show()

lang_heatmap = title_language_df.pivot(columns='source', index='target')
lang_heatmap.columns = [x[1] for x in lang_heatmap.columns]
fig, axs = plt.subplots(figsize=(15, 9))
sns.heatmap(lang_heatmap.T, annot=True, cmap='Blues', fmt='.2%', ax=axs)

task_heatmap = title_task_df.pivot(columns='source', index='target')
task_heatmap.columns = [x[1] for x in task_heatmap.columns]
task_heatmap.index = [
    'Analyse data',
    'Build ML services',
    'Build data infra',
    'Build ML prototypes',
    'Research',
    'Improve ML models',
    'None'
]
sns.heatmap(task_heatmap.T, annot=True, cmap='Blues')


warnings.filterwarnings('ignore')

selected_langs = ['Python', 'R', 'SQL']
fig = px.line_polar(
    (
        title_language_df
        .query('target in @selected_langs')
    ),
    r='value',
    range_r=[0, 1],
    color='target',
    theta='source',
    line_close=True,
    color_discrete_sequence=px.colors.qualitative.G10
)

fig.update_layout(
    legend_title_text='Programming Language',
    autosize=False,
    width=1000,
    height=400,
    margin=dict(
        t=100,
    ),
    title='<b>Use of technologies by title</b>',
    title_font_size=20,
    title_x=0.5,
    title_font_family='arial',
    title_font_color='#C41E3A',
)

fig.write_image('./charts/Use of technologies1.png', scale=2)

df['ml_algo_Neural Networks'] = df[[
    'ml_algo_Dense Neural Networks (MLPs, etc)',
    'ml_algo_Convolutional Neural Networks',
    'ml_algo_Generative Adversarial Networks',
    'ml_algo_Recurrent Neural Networks',
    'ml_algo_Transformer Networks (BERT, gpt, etc)',
    'ml_algo_Autoencoder Networks (DAE, VAE, etc)',
    'ml_algo_Graph Neural Networks',
]].any(axis=1).replace(False, np.nan)

df['ml_algo_trees'] = df[[
    'ml_algo_Decision Trees or Random Forests',
    'ml_algo_Gradient Boosting Machines (xgboost, lightgbm, etc)',
]].any(axis=1).replace(False, np.nan)

algo_mapping = {
    'ml_algo_Linear or Logistic Regression': 'Linear/Logistic Regression',
    'ml_algo_Decision Trees or Random Forests': 'Decision Trees/Random Forests',
    'ml_algo_Gradient Boosting Machines (xgboost, lightgbm, etc)': 'Gradient Boosting Machines',
    'ml_algo_Neural Networks': 'Neural Networks',
}

title_algo = []
for title in titles:
    for algo in algo_mapping.keys():
        title_algo.append([
            title,
            algo_mapping[algo],
            df.query('title == @title').dropna(subset=[algo]).shape[0] / df.query('title == @title').shape[0],
        ])

title_algo_df = pd.DataFrame(title_algo, columns=['title', 'Algorithm', 'value'])

fig2 = px.line_polar(
    title_algo_df,
    r='value',
    color='Algorithm',
    theta='title',
    line_close=True,
    color_discrete_sequence=px.colors.qualitative.G10
)

fig2.update_layout(
    autosize=False,
    width=1000,
    height=400,
)

fig2.write_image('./charts/Use of technologies2.png', scale=2)

cloud_mapping = {
    'cloud_platform_AmazonWebServices(AWS)': 'AWS',
    'cloud_platform_MicrosoftAzure': 'Azure',
    'cloud_platform_GoogleCloudPlatform(GCP)': 'GCP',
}

title_cloud_df = []
for title in titles:
    for cloud in cloud_mapping.keys():
        title_cloud_df.append([
            title,
            cloud_mapping[cloud],
            df.query('title == @title').dropna(subset=[cloud]).shape[0] / df.query('title == @title').shape[0],
        ])

title_cloud_df = pd.DataFrame(title_cloud_df, columns=['title', 'Cloud Platform', 'value'])


fig3 = px.line_polar(
    title_cloud_df,
    r='value',
    color='Cloud Platform',
    theta='title',
    line_close=True,
    color_discrete_sequence=px.colors.qualitative.G10
)

fig3.update_layout(
    autosize=False,
    width=1000,
    height=400,
)

fig3.write_image('./charts/Use of technologies3.png', scale=2)
