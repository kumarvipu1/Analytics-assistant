from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.express as px
import streamlit as st
import re
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import pandas as pd
import os
import logging




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

# Function to save user form data to BigQuery
def save_user_form_to_bigquery(firstname, lastname, email, occupation, company, phone):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Replace 'path/to/your/service_account.json' with the actual path to your service account key file
        key_path = '/Users/Mishael.Ralph-Gbobo/PycharmProjects/Analytics-assistant/service_account.json'
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = bigquery.Client(credentials=credentials, project='analyticsassistantproject')

        dataset_id = 'user_forms'  # Corrected dataset ID
        table_id = 'user_forms'  # Corrected table ID

        # Construct the table reference
        table_ref = client.dataset(dataset_id).table(table_id)

        # Get the table to ensure it exists
        table = client.get_table(table_ref)

        # Define the rows to insert
        rows_to_insert = [{
            "firstname": firstname,
            "lastname": lastname,
            "email": email,
            "occupation": occupation,
            "company": company,
            "phone": phone
        }]

        # Insert rows
        errors = client.insert_rows_json(table, rows_to_insert)

        if not errors:
            logger.info("New rows have been added.")
        else:
            logger.error(f"Encountered errors while inserting rows: {errors}")

    except Exception as e:
        logger.error(f"Exception occurred: {e}")