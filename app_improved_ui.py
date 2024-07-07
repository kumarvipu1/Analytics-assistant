import pandas as pd
import matplotlib
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_prompt import main_prompt
import contextlib
import sys
import io
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from prompt import instruction_st, new_prompt, context
from helper_tools import graph_plot_engine, parse_response
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go

matplotlib.use('agg')

# Page configuration
st.set_page_config(
    page_title="Auto Analyst App",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)

api_key = None

@contextlib.contextmanager
def capture_output():
    new_out = io.StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def process_data(df, selected_x, selected_y, selected_z):
    grouped_df = df.groupby([selected_x, selected_z]).agg({selected_y: 'mean'}).reset_index()
    sorted_df = grouped_df.sort_values(by=selected_y, ascending=False)
    top_10_x = sorted_df[selected_x].head(10).tolist()
    filtered_df = grouped_df[grouped_df[selected_x].isin(top_10_x)]
    return filtered_df

def main():
    st.sidebar.title('Upload Data')
    st.sidebar.write('**Upload your data here 👇**')

    st.title('Your Personal Data Analyst!')

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here")

    if uploaded_file:
        df = load_data(uploaded_file)
        categorical_vars = list(df.select_dtypes(include=['object', 'category']).columns)
        numeric_vars = list(df.select_dtypes(include=['int64', 'float64']).columns)

        query_agent = PandasQueryEngine(df=df, verbose=False, instruction_srt=instruction_st)
        query_agent.update_prompts({"pandas_prompt": new_prompt})

        tools = [QueryEngineTool(query_engine=query_agent, metadata=ToolMetadata(
            name="pandas_query_agent",
            description="used for query pandas dataframe for data analytics needs"
        ))]

        if api_key:
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo-0613', openai_api_key=api_key),
                df, verbose=False,
                agent_type=AgentType.OPENAI_FUNCTIONS
            )
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)
        else:
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo-0613'),
                df, verbose=False,
                agent_type=AgentType.OPENAI_FUNCTIONS
            )
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)

        llm_agent = ReActAgent.from_tools(tools, llm=llm, verbose=False, context=context)
        exp_result = llm_agent.query('explain me like five each column in the dataset df in bullet points')
        exp_output = str(exp_result)

        st.write("### Data Preview")
        st.write(df.head())

        with st.expander("See explanation"):
            st.write(exp_output)

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            st.markdown("### Visualization 1")
            selected_x1 = st.selectbox('Select X', categorical_vars)
            selected_y1 = st.selectbox('Select Y', numeric_vars)
            selected_z1 = st.selectbox('Select legends', categorical_vars)

            if selected_x1 != selected_z1:
                filtered_df1 = process_data(df, selected_x1, selected_y1, selected_z1)
                fig1 = px.bar(filtered_df1, x=selected_x1, y=selected_y1, color=selected_z1,
                              title='Plot 1',
                              labels={selected_x1: 'X', selected_y1: 'Mean of User Score', selected_z1: 'Legends'},
                              color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig1)

        with fig_col2:
            st.markdown("### Visualization 2")
            selected_x2 = st.selectbox('Select X', categorical_vars, key='a1')
            selected_y2 = st.selectbox('Select Y', numeric_vars, key='b1')
            selected_z2 = st.selectbox('Select legends', categorical_vars, key='c1')

            if selected_x2 != selected_z2:
                filtered_df2 = process_data(df, selected_x2, selected_y2, selected_z2)
                fig2 = px.bar(filtered_df2, x=selected_x2, y=selected_y2, color=selected_z2,
                              title='Plot 2',
                              labels={selected_x2: 'X', selected_y2: 'Mean of User Score', selected_z2: 'Legends'},
                              color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig2)

        st.markdown("### Query Your Data")
        query = st.text_area("Insert your query")

        # Store outputs in session state to persist across reruns
        if 'query_outputs' not in st.session_state:
            st.session_state['query_outputs'] = []

        if st.button("Submit Query"):
            result = agent.invoke(main_prompt.format(query_str=query))
            code, text = parse_response(result['output'])

            if ('plt' in code) or ('px' in code):
                exec(code)
                st.pyplot()
            else:
                with capture_output() as captured:
                    exec(code)
                    output = captured.getvalue() + '\n'
                st.session_state.query_outputs.append(output)
                st.write(output)

        # Display all past query outputs
        if st.session_state['query_outputs']:
            st.markdown("### Previous Outputs")
            for output in st.session_state['query_outputs']:
                st.write(output)

if __name__ == "__main__":
    main()
