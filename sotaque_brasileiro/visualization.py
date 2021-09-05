import pandas as pd
import plotly.express as px

from sotaque_brasileiro.constants import constants


def plot_geo_heatmap(df: pd.DataFrame, columns_set: str):
    """
    Plot a heatmap of a given set of columns in a dataframe.
    """
    if columns_set not in ["birth", "current", "parents_original"]:
        raise ValueError(
            "columns_set must be one of 'birth', 'current', 'parents_original'"
        )
    match_columns_set = {
        "birth": "Mapa de participações por cidade de nascimento",
        "current": "Mapa de participações por cidade atual",
        "parents_original": "Mapa de participações por cidade de origem dos pais",
    }
    df = (
        df.groupby([f"{columns_set}_latitude", f"{columns_set}_longitude"])
        .id.nunique()
        .reset_index(name="count")
    )
    fig = px.density_mapbox(
        df,
        lat=f"{columns_set}_latitude",
        lon=f"{columns_set}_longitude",
        z="count",
        zoom=4,
        center=constants.BRAZIL_CENTER_DEFAULT.value,
        radius=40,
        mapbox_style="stamen-terrain",
        title=match_columns_set[columns_set],
    )
    fig.show()


def plot_engagement_over_time(df: pd.DataFrame):
    """
    Plot line charts containing number of audios over time.
    """
    df = df.groupby("date")["id"].count().reset_index()
    df["cumulative"] = df["id"].cumsum()
    fig = px.line(
        df, x="date", y="cumulative", title="Crescimento da Sotaque Brasileiro"
    )
    fig.show()
