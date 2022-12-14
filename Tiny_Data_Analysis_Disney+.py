# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: 2022F DATA5006 Computer Programming in Python
#     language: python
#     name: 2022f_data5006
# ---

# %% [markdown]
# # About This Colab Notebook
#
# 1. This Colab is designed for the course "**Computer Programming in Python**" instructed by Tse-Yu Lin.
#
# 2. Note that each time you enter this Colab from the link provided by the instructor, TAs or other people.
# Please create a copy to your own Google Drive by clicking **File > Save a copy in Drive**. After clicking, a new tab page will be opened.

# %%
# %matplotlib widget

import calendar
import json
import os
from itertools import combinations
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.text import Text

sns.set_theme(palette="pastel")

# %% [markdown]
# # Preparation: Download Dataset

# %% [markdown]
# We import the following package ``gdown`` to download files from Google Drive.

# %%
filename = "disney_plus_titles.csv"
url = "https://drive.google.com/u/1/uc?id=118sEZB_-OfyXH130f-EED4KI26AXH8FO&export=download"
# mirror_url = "https://drive.google.com/u/1/uc?id=1xlzqH8z2ChAO4xPpm3GDfslyni53Bn5j&export=download"
try:
    if "google.colab" in str(get_ipython()):  # type: ignore
        import gdown  # type: ignore

        gdown.download(url, filename)
except NameError:
    pass

# %%
raw_df = pd.read_csv(filename)

# %%
m, n = raw_df.shape
print(f"Size of dataset: {m}")
print(f"Number of Features: {n}")

# %%
raw_df.sample(n=5, random_state=42)

# %% [markdown]
# Please refer the following link for more detail about headers:
# * Disney+ Movies and TV Shows: https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows

# %% [markdown]
# # Data Analysis on Your Own

# %%
raw_df.info()


# %% [markdown]
# ## Define Utility Functions

# %%
def rotate_label(
    label: Text,
    rotation: float | Literal["vertical", "horizontal"],
    align: Literal["left", "center", "right"] = "right",
) -> None:
    """Rotates a given `Text` instance by `rotation` degrees.

    Args:
        label: The `Text` instance to rotate.
        rotation: The rotation in degrees.
        align: The horizontal alignment of the resulting object. Defaults to "right".
    """
    label.set_rotation(rotation)
    label.set_horizontalalignment(align)


# %%
def to_edges(cast: list) -> list:
    """Converts a list of casts into node edges between the casts.

    For example:
    If the function is called with the list [A, B, C], then the edges produced are:
    * A -- B
    * A -- C
    * B -- C
    >>> to_edges(["A", "B", "C"])
    [('A', 'B'), ('A', 'C'), ('B', 'C')]

    Args:
        cast: The list of casts between which edges are to be created

    Returns:
        The list of cast edges.
    """
    return list(combinations(sorted(set(cast)), 2))


# %% [markdown]
# ## Clean the Data
#
# - drop the `directors` column as it contains a substantial amount of `NA` values
# - drop the `description` column as it will not participate in the analysis
# - convert the `dtype` of the `date` column into `datetime`
# - extract only the numbers in the `duration` column and convert its `dtype` to `int`
# - split the comma-separated string of casts in the `cast` column into a list of casts
# - ditto for `country` and `listed_in` columns
# - set the `type`, `rating`, `release_year`, `date_added`, and `show_id` columns as the indices
# - sort the indices in ascending order

# %%
cleaned_df = (
    raw_df.drop(labels=["director", "description"], axis="columns")
    .dropna(subset=["rating", "date_added"], how="any", axis="index")
    .assign(
        date_added=pd.to_datetime(raw_df["date_added"], format="%B %d, %Y"),
        duration=raw_df["duration"].str.extract("([0-9]+)").astype(int),
        cast=raw_df["cast"].str.split(", "),
        country=raw_df["country"].str.split(", "),
        listed_in=raw_df["listed_in"].str.split(", "),
    )
    .set_index(keys=["type", "rating", "release_year", "date_added", "show_id"])
    .sort_index()
)

cleaned_df.info()
cleaned_df  # type: ignore

# %% [markdown]
# ## Plot Features Against Counts

# %%
# count the number of shows for each show type, i.e. Movie, TV Show
data = cleaned_df["title"].groupby(level="type").count().sort_values(ascending=False)
fig, ax = plt.subplots()  # create a figure with one axes to plot the data
# plot the data, i.e. as a pie chart
ax.pie(
    x=data,
    labels=data.index.get_level_values("type").tolist(),  # set values of type as labels
    autopct=lambda x: int(x / 100 * data.sum()),  # show absolute values of each slice
)
ax.set_title("Show Types")
# make the axes layout as tight to the borders of the figure as possible
_ = fig.tight_layout()

# %% [markdown]
# Around 2/3 of show types in Disney+ are Movies, while the rest are TV Shows.

# %%
fig, ax = plt.subplots()
sns.countplot(
    cleaned_df,
    x=cleaned_df.index.get_level_values("rating"),  # set values of rating as x-axis
    ax=ax,  # plot the data on the axes `ax`
)
ax.set_title("Rating vs. Count")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
# rotate each label in the plot's x-tick-labels by 30 degrees
{rotate_label(label, rotation=30) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# Most Disney+ shows are suitable for all ages (G & TV-G), some might be unsuitable for younger children (PG & TV-PG).

# %%
fig, ax = plt.subplots()
sns.countplot(
    cleaned_df,
    x=cleaned_df.index.get_level_values("rating"),
    hue=cleaned_df.index.get_level_values("type"),  # set values of type as bar colors
    ax=ax,
)
ax.set_title("Rating vs. Type vs. Count")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
_ = fig.tight_layout()

# %% [markdown]
# In general, the only ratings assigned to TV shows are ones that look like TV-*,
# while movies may be assigned any rating (e.g. PG & TV-PG).
#
# However, there is one exception, shown below.

# %%
# get all rows where `rating` does not start with `TV` and `type` is `TV Show`
cleaned_df[
    ~(cleaned_df.index.get_level_values("rating").str.startswith("TV"))
    & (cleaned_df.index.get_level_values("type") == "TV Show")
]

# %%
fig, ax = plt.subplots()
sns.histplot(data=cleaned_df, x="release_year", ax=ax)
ax.set_title("Release Year vs. Count")
ax.set_xlabel("Release Year")
ax.set_ylabel("Count")
{rotate_label(label, rotation=30) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# Most of the shows added by Disney+ are released in recent decades.

# %%
# create a figure with 1 row, 2 columns and a size of 9.6 in x 4.8 in to plot the data
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.8))
sns.violinplot(
    x=cleaned_df.index.get_level_values("type"),
    y=cleaned_df.index.get_level_values("release_year"),
    scale="width",  # make all violins have the same width
    ax=axs[0],
)
axs[0].set_title("Release Year vs. Type")
axs[0].set_xlabel("Type")
axs[0].set_ylabel("Release Year")
p = sns.histplot(
    cleaned_df,
    x=cleaned_df.index.get_level_values("release_year"),
    hue=cleaned_df.index.get_level_values("type"),
    kde=True,  # include the kernel density estimate in the plot
    ax=axs[1],
)
p.legend_.set_title("Type")
axs[1].set_title("Release Year vs. Type vs. Count")
axs[1].set_xlabel("Release Year")
axs[1].set_ylabel("Count")
_ = fig.tight_layout()

# %% [markdown]
# The width of violin plot shows how many Disney+ shows are released each year. The wider the body of the 'violin', the more Disney+ shows are released in the particular year. As seen above, most movies and TV shows are released within year 2000-2020.

# %%
# sort cleaned_df by `date_added` only
data = cleaned_df.sort_index(level="date_added", sort_remaining=False)
# get the list of `date_added` in the format "yyyy-mm"
date_added_df = data.index.get_level_values("date_added").strftime("%Y-%m")  # type: ignore
fig, ax = plt.subplots()
sns.countplot(
    x=date_added_df.sort_values(),
    color=sns.color_palette()[0],  # type: ignore # set the color of the bars
    width=1.0,  # set the width of each bar to 1.0
    ax=ax,
)
ax.set_title("Date Added vs. Count")
ax.set_xlabel("Date Added")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[0])  # type: ignore
ax.set_yscale("log")
{rotate_label(label, rotation=45) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# Huge amount of Disney+ shows were added on November '19. According to Wikipedia, Disney+ was officially launched on November '19.

# %%
fig, ax = plt.subplots(figsize=(9.6, 4.8))
p = sns.histplot(
    data,
    x=date_added_df,
    hue=data.index.get_level_values("type"),
    discrete=True,  # set bin widths as 1.0 and integers as the center of each bin
    kde=True,
    log_scale=(0, 10),  # plot the y-axis in log scale
    ax=ax,
)
p.legend_.set_title("Type")
ax.set_title("Date Added vs. Type vs. Count")
ax.set_xlabel("Date Added")
ax.set_ylabel("Count")
{rotate_label(label, rotation=30) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# Of all the Disney+ shows that were added between 2019 until 2021, there are slightly more movies added compared to TV shows.

# %%
# create a new column, `month_added`, with the list of `date_added` in the format "mm" and sort by `month_added`
data = cleaned_df.assign(
    month_added=cleaned_df.index.get_level_values("date_added").strftime("%m")  # type: ignore
).sort_values("month_added")
fig, ax = plt.subplots()
p = sns.histplot(
    data,
    x="month_added",
    discrete=True,
    ax=ax,
)
ax.set_title("Month Added vs. Count")
ax.set_xlabel("Month Added")
ax.set_ylabel("Count")
ax.set_xticklabels(list(calendar.month_name[1:]))  # convert month number to month name
{rotate_label(label, rotation=45) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# The above plot is not truly representative of the distribution as Disney+ was officialy launched on November which skews the distribution immensely.

# %%
fig, ax = plt.subplots()
p = sns.histplot(
    data,
    x="month_added",
    hue=data.index.get_level_values("type"),
    discrete=True,
    log_scale=(0, 10),
    ax=ax,
)
p.legend_.set_title("Type")
ax.set_title("Month Added vs. Type vs. Count")
ax.set_xlabel("Month Added")
ax.set_ylabel("Count")
ax.set_xticklabels(list(calendar.month_name[1:]))
{rotate_label(label, rotation=45) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# Of the Disney+ shows added in the launch date November '19, more movies were added compared to TV shows.

# %%
# get all rows where the `type` of show is `Movie`
data = cleaned_df[cleaned_df.index.get_level_values("type") == "Movie"]
fig, ax = plt.subplots()
ax = sns.histplot(data, x="duration", bins=185 // 5, binrange=(0, 185))  # type: ignore
ax.set_title("Movie Duration vs. Count")
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Count")
_ = fig.tight_layout()

# %%
(
    len(data[(5 <= data["duration"]) & (data["duration"] < 10)]),
    len(data[(45 <= data["duration"]) & (data["duration"] < 50)]),
    len(data[(90 <= data["duration"]) & (data["duration"] < 95)]),
)  # type: ignore

# %% [markdown]
# There are three peaks in the distribution; 150, 62, and 120 shows have a duration of 10-mins, 45-mins, and 90-mins respectively.

# %%
# get all rows where the `type` of show is `TV Show`
data = cleaned_df[cleaned_df.index.get_level_values("type") == "TV Show"]
fig, ax = plt.subplots()
# plot the data, set any values larger than 12 to 12 so as not to draw an extended tail
ax = sns.histplot(x=data["duration"].clip(0, 12), discrete=True, log_scale=(0, 10))
ax.set_title("TV Show Seasons vs. Count")
ax.set_xlabel("Number of Seasons")
ax.set_ylabel("Count")
ax.bar_label(ax.containers[1])

labels: list[Text] = ax.get_xticklabels()  # get the current `xticklabels`
labels[-2].set_text("12+")  # modify the label "12" to "12+"
ax.set_xticklabels(labels)  # set the `xticklabels` to the new labels
_ = fig.tight_layout()

# %%
data[data["duration"] >= 12].sort_values("duration")

# %%
(
    len(data[data["duration"] == 1]) / len(data),
    len(data[data["duration"] > 5]) / len(data),
)  # type: ignore

# %% [markdown]
# More than half of TV Shows only last for one season, while less than 5% of TV Shows last for more than 5 seasons.

# %%
# get the list of countries, dropping `NA` values and sorting the resulting values
data = cleaned_df["country"].dropna().explode().sort_values()
fig, ax = plt.subplots(figsize=(9.6, 4.8))
sns.countplot(x=data, ax=ax)
ax.set_title("Country vs. Count")
ax.set_xlabel("Country")
ax.set_ylabel("Count")
ax.set_yscale("log")
{rotate_label(label, 45) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# Most Disney+ shows are added in english-speaking countries such as the US, the UK, Canada, and Australia.

# %%
# ditto for genres
data = cleaned_df["listed_in"].explode().sort_values()
fig, ax = plt.subplots(figsize=(9.6, 4.8))
sns.countplot(x=data, ax=ax)
ax.set_title("Genre vs. Count")
ax.set_xlabel("Genre")
ax.set_ylabel("Count")
ax.set_yscale("log")
{rotate_label(label, 45) for label in ax.get_xticklabels()}
_ = fig.tight_layout()

# %% [markdown]
# The various genres of Movies and TV Shows are listed in the bar plot above with the frequency of each shown.

# %%
data = cleaned_df["title"].str.len()  # get the list of title character lengths
fig, ax = plt.subplots()
ax = sns.histplot(x=data)
ax.set_title("Title Length vs. Count")
ax.set_xlabel("Title Length (characters)")
ax.set_ylabel("Count")
_ = fig.tight_layout()

# %% [markdown]
# The figure shows the frequency of title character length, it can be seen that titles with 15 to 17 characters has the highest counts.

# %%
# get the list of title word lengths
data = cleaned_df["title"].str.findall(r"\w+").apply(len)
fig, ax = plt.subplots()
ax = sns.histplot(x=data, discrete=True)
ax.set_title("Title Length vs. Count")
ax.set_xlabel("Title Length (words)")
ax.set_ylabel("Count")
_ = fig.tight_layout()

# %% [markdown]
# The figure shows the frequency of title word counts, and it can be observed that the highest word count is 3.

# %%
data = (
    cleaned_df["cast"]
    .dropna()
    .apply(set)  # type: ignore # convert each list of casts to a set of casts
    .explode()
    .to_frame("cast")  # convert the Series into a DataFrame
    .groupby("cast")
    .apply(len)  # count the number of movies played by each cast
    .value_counts()  # count the frequency of each number of movies played
)
fig, ax = plt.subplots()
ax = sns.lineplot(data)
ax.set_title("Movies Played vs. Cast Count")
ax.set_xlabel("Number of Movies Played")
ax.set_ylabel("Count")
_ = fig.tight_layout()

# %%
data.loc[1] / data.sum()  # type: ignore

# %% [markdown]
# The figure shows the frequency of casts with the specified number of movies played; 73.0% casts starring in Disney+ shows only stars in one movie,

# %% [markdown]
# ### Cast Connectivity Graph

# %%
# find all edges between every cast and every other cast
edges_df: pd.DataFrame = (
    cleaned_df["cast"]
    .dropna()
    .reset_index(drop=True)
    .apply(to_edges)
    .explode(ignore_index=True)  # type: ignore
    .dropna()
    .apply(pd.Series)
    .rename(columns={i: v for i, v in enumerate(("source", "target"))})
    .convert_dtypes()
)
# set the weight of each edge as the number of occurrences of said edge
weighted_edges_df = pd.merge(
    edges_df.drop_duplicates(),
    edges_df.value_counts().rename("weight"),
    left_on=("source", "target"),
    right_index=True,
).sort_values("weight", ascending=False)
# create a graph from the weighted edges
G = nx.from_pandas_edgelist(weighted_edges_df, edge_attr=True)
# extract the largest subgraph
g = G.subgraph(max(nx.connected_components(G), key=len))

# %%
# check whether node positions are available
if "cast_graph_positions.json" not in os.listdir():

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return {"__numpy__": list(obj)}
            return json.JSONEncoder.default(self, obj)

    # calculate node plotting positions
    pos = pd.DataFrame(
        nx.spring_layout(g, k=1 / (2 * np.sqrt(len(g))), iterations=1000, seed=42)
    )
    # save the results in a json file
    with open("cast_graph_positions.json", "w") as f:
        json.dump(pos, f, cls=NumpyEncoder)
else:

    def as_numpy(dct: dict) -> np.ndarray | dict:
        if "__numpy__" in dct:
            return np.array(dct["__numpy__"])
        return dct

    # load node plotting positions
    with open("cast_graph_positions.json") as f:
        pos = json.load(f, object_hook=as_numpy)

pos_df = pd.DataFrame(pos).T  # convert node positions into a DataFrame

# %%
# set the "Jim Cummings" node as the origin and rescale distances
transformed_pos_df = pos_df.sub(pos_df.loc["Jim Cummings"]).apply(
    lambda x: x / (np.dot(x, x) ** 0.2 + 1e-8), axis="columns"
)

# get the degree of each cast and sort the values by degree
node_degree = pd.DataFrame(g.degree, columns=["cast", "degree"]).sort_values("degree")

# plot the network
fig, ax = plt.subplots(figsize=(16, 9))
nx.draw_networkx(
    g,
    pos=transformed_pos_df.T.to_dict(orient="list"),
    ax=ax,
    alpha=0.9,
    cmap="rainbow",
    edge_color="white",
    node_color=[v for v in node_degree.loc[:, "degree"]],
    node_size=[v**1.5 for v in node_degree.loc[:, "degree"]],
    nodelist=node_degree.loc[:, "cast"],
    width=0.1,
    with_labels=False,
)
ax.axis("off")
ax.set_title("Cast Connectivity Graph", color="white", size=24)
fig.patch.set_facecolor("#131327")  # type: ignore  # set the figure background color
fig.tight_layout()
plt.savefig("test.svg", format="svg")

# %% [markdown]
# The cast connectivity graph shows how casts in Disney+ shows are connected to each other.
# Each nodes represents one cast, casts with bigger nodes means they have played with more casts compared to the smaller nodes.
# The red node (which is the biggest one) is Jim Cummings, the official voice of Winnie the Pooh, who has appeared in almost 400 roles.
