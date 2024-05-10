from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
import plotly.express as px
import streamlit as st
import re
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import pandas as pd
import os



@st.cache_data
def plot_graph(df, x_col, y_col, type):
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
    if type == 'line':
        fig = px.line(df, x=x_col, y=y_col)
    elif type =='scatter':
        fig = px.scatter(df, x=x_col, y=y_col)
    elif type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col)
    elif type == 'histogram':
        fig = px.histogram(df, x=x_col, y=y_col)
    else:
        raise ValueError(f"Invalid graph type: {type}. Supported types are 'line','scatter', 'bar', 'histogram'")

    fig.show()


def parse_response(response):
    decoded = bytes(str(response), "utf-8").decode("unicode_escape")
    pattern = r'```python\s*([\s\S]*?)\s*```'

    matches = re.findall(pattern, decoded)
    if matches:
        code = str(matches[0])
        text = re.sub(re.escape(code), "", decoded).strip()
    else:
        code = "print("")"
        text = decoded

    return code, text


graph_plot_engine = FunctionTool.from_defaults(
    fn=plot_graph,
    tool_metadata=ToolMetadata(
    name="graph",
    description="Plots a suitable type of graph based on requirement and column input, the options to pick from are 'line','scatter', 'bar', 'histogram'",)

    )

def save_user_form(user_form_df, firstname, lastname, email, occupation, company, phone):
    file_path = 'user_forms.csv'
    
    new_entry = {
        'firstname': firstname,
        'lastname': lastname,
        'email': email,
        'occupation': occupation,
        'company': company,
        'Phone': phone
    }

    user_form_df = pd.DataFrame([new_entry])
    file_exists = os.path.exists(file_path)
    user_form_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return None