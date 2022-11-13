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
# # 0. About This Colab Notebook
#
# 1. This Colab is designed for the course "**Computer Programming in Python**" instructed by Tse-Yu Lin.
#
# 2. Note that each time you enter this Colab from the link provided by the instructor, TAs or other people.
# Please create a copy to your own Google Drive by clicking **File > Save a copy in Drive**. After clicking, a new tab page will be opened.

# %%
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.text import Text

sns.set_theme(palette="pastel")

# %% [markdown]
# # a) Preparation: Download Dataset

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
# # b) Data Analysis on Your Own

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
    label.set_ha(align)


# %% [markdown]
# ---

# %% [markdown]
# ## Clean the Data

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
    .convert_dtypes()
    .set_index(keys=["type", "rating", "release_year", "date_added", "show_id"])
    .sort_index()
)

cleaned_df.info()
cleaned_df

# %% [markdown]
# ---

# %% [markdown]
# ## Plot Features Against Counts

# %%
data = cleaned_df["title"].groupby(level="type").count().sort_values(ascending=False)
plt.pie(
    x=data,
    labels=data.index.get_level_values("type"),
    autopct=lambda x: int(x / 100 * data.sum()),
)
_ = plt.title("Show Types")

# %%
p = sns.countplot(cleaned_df, x=cleaned_df.index.get_level_values("rating"))
p.set_title("Rating vs. Count")
p.set_xlabel("Rating")
p.set_ylabel("Count")
_ = {rotate_label(label, rotation=30) for label in p.get_xticklabels()}

# %%
p: plt.Axes = sns.histplot(data=cleaned_df, x="release_year")
p.set_title("Release Year vs. Count")
p.set_xlabel("Release Year")
_ = p.set_ylabel("Count")

# %%
data = cleaned_df.index.get_level_values("date_added").strftime("%Y-%m").sort_values()
p = sns.countplot(
    x=data.str.extract("([0-9]+)").loc[:, 0],
    color=sns.color_palette()[0],
    saturation=0.75,
    width=1.0,
)
p.set_title("Year Added vs. Count")
p.set_xlabel("Year Added")
_ = p.set_ylabel("Count")

# %%
p: plt.Axes = sns.countplot(
    x=data, color=sns.color_palette()[0], saturation=0.75, width=1.0
)
p.set_title("Date Added vs. Count")
p.set_xlabel("Date Added")
p.set_ylabel("Count")
p.bar_label(p.containers[0])
p.set_yscale("log")
_ = {rotate_label(label, rotation=60) for label in p.get_xticklabels()}

# %%
data = cleaned_df[cleaned_df.index.get_level_values("type") == "Movie"]
p = sns.histplot(data, x="duration", bins=185 // 5, binrange=(0, 185))
p.set_title("Movie Duration vs. Count")
p.set_xlabel("Duration (minutes)")
_ = p.set_ylabel("Count")

# %%
data = cleaned_df[cleaned_df.index.get_level_values("type") == "TV Show"]
p = sns.histplot(x=data["duration"].clip(0, 16), discrete=True, log_scale=(0, 10))
p.set_title("TV Show Seasons vs. Count")
p.set_xlabel("Number of Seasons")
p.set_ylabel("Count")
p.bar_label(p.containers[1], labels=p.containers[1].datavalues)

labels: list[Text] = p.get_xticklabels()
labels[-2] = "16+"
_ = p.set_xticklabels(labels)

# %% [markdown]
# Ignore NA values for country

# %%
data = cleaned_df["country"].dropna().explode().sort_values()
fig, ax = plt.subplots(figsize=(9.6, 4.8))
sns.countplot(x=data, ax=ax)
ax.set_title("Country vs. Count")
ax.set_xlabel("Country")
ax.set_ylabel("Count")
ax.set_yscale("log")
_ = {rotate_label(label, 60) for label in ax.get_xticklabels()}

# %%
