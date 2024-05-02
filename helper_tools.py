from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
import plotly.express as px
import streamlit as st
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import pandas as pd



@st.cache_data
def plot_graph(df, x_col, y_col, graph_type):
    """
    Plot a Plotly Express graph based on the type of graph requested.

    Parameters:
    df (pandas DataFrame): Dataframe containing the data to plot
    x_col (str): Column name for the x-axis
    y_col (str): Column name for the y-axis
    graph_type (str): Type of graph to plot (e.g. 'line','scatter', 'bar', 'histogram')

    Returns:
    None
    """
    if graph_type == 'line':
        fig = px.line(df, x=x_col, y=y_col)
    elif graph_type =='scatter':
        fig = px.scatter(df, x=x_col, y=y_col)
    elif graph_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col)
    elif graph_type == 'histogram':
        fig = px.histogram(df, x=x_col, y=y_col)
    else:
        raise ValueError(f"Invalid graph type: {graph_type}. Supported types are 'line','scatter', 'bar', 'histogram'")

    fig.show()


def get_schema(df, column_list):
    """

    :param df: main dataframe (pandas DataFrame)
    :param column_list: list of columns of interest
    :return:
    """
    print('just a placeholder')


graph_plot_engine = FunctionTool.from_defaults(
    fn=plot_graph,
    tool_metadata=ToolMetadata(
    name="graph plotter",
    description="Plots a suitable type of graph based on requirement and column input, the options to pick from are 'line','scatter', 'bar', 'histogram'",)

    )
