import streamlit as st
import pandas as pd
import llama_index
import matplotlib.pyplot as plt
from llama_index.experimental.query_engine import PandasQueryEngine
from helper_tools import graph_plot_engine, parse_response
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from prompt import instruction_st, new_prompt, context
import contextlib
import sys
import io
import numpy as np
import plotly.express as px
import altair as alt
import json
import openai
import json
from datetime import date


# Setting up agent

# Page configuration
st.set_page_config(
    page_title="Auto Analyst App",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)

api_key = None  # Replace None with your api key

@contextlib.contextmanager
def capture_output():
    new_out = io.StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out



def main():

    # create sidebar
    st.sidebar.write('**Upload your data here 👇**')

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here")

    if uploaded_file:

        # Setting up llm agent
        df = pd.read_csv(uploaded_file)
        query_agent = PandasQueryEngine(df=df, verbose=True, instruction_srt=instruction_st)

        query_agent.update_prompts({"pandas_prompt": new_prompt})

        tools = [QueryEngineTool(query_engine=query_agent,
                                 metadata=ToolMetadata(
                                     name="pandas_query_agent",
                                     description="used for query pandas dataframe for data analytics needs"),
                                 ),
                 ]

        if api_key:
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)
        else:
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)

        llm_agent = ReActAgent.from_tools(tools, llm=llm, verbose=True,
                                          context=context)

        # Building Expander output

        exp_result = llm_agent.query('explain me like five each column in this dataset in bullet points')
        exp_output = str(exp_result)

        st.title('Your Personal Data Analyst!')

        df_sum = pd.DataFrame(df.head())
        st.write("### Data Preview")
        st.write(df_sum)

        with st.expander("See explanation"):
            st.write(exp_output)

        col = st.columns((10, 15, 2), gap='medium')

        with col[0]:
            st.markdown('#### Visualization 1')

            result = llm_agent.query('provide python code to visualize the most insightful categorical variable and one \
            of the numeric variable from this data')

            code, text = parse_response(result)

            if 'plt' in code:
                fig = exec(code)
                st.pyplot(fig)

            else:

                with capture_output() as captured:
                    exec(code)
                    output = captured.getvalue() + '\n'
                st.write(output)



if __name__ == "__main__":
    main()