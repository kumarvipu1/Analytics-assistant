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


def main():
    # create sidebar
    st.sidebar.write('**Upload your data here 👇**')

    st.title('Your Personal Data Analyst!')

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here")

    if uploaded_file:

        # Setting up llm agent
        df = pd.read_csv(uploaded_file)

        # Get all categorical variables
        categorical_vars = list(df.select_dtypes(include=['object', 'category']).columns)

        # Get all categorical variables
        numeric_vars = list(df.select_dtypes(include=['int64', 'float64']).columns)

        # initializing tools for llama_index agent
        query_agent = PandasQueryEngine(df=df, verbose=False, instruction_srt=instruction_st)

        query_agent.update_prompts({"pandas_prompt": new_prompt})

        tools = [QueryEngineTool(query_engine=query_agent,
                                 metadata=ToolMetadata(
                                     name="pandas_query_agent",
                                     description="used for query pandas dataframe for data analytics needs"),
                                 ),
                 ]

        if api_key:
            # langchain agent
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo-0613',
                           openai_api_key=api_key),
                df, verbose=False,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True
            )

            # llama_index agent
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)

        else:

            # langchain_agent
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0613'),
                df, verbose=False,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True
            )

            # llama_index agent
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)

        # llama_index agent with tools
        llm_agent = ReActAgent.from_tools(tools, llm=llm, verbose=False,
                                          context=context)

        exp_result = llm_agent.query('explain me like five each column in the dataset df in bullet points')
        exp_output = str(exp_result)

        df_sum = pd.DataFrame(df.head())
        st.write("### Data Preview")
        st.write(df_sum)

        # explanation of dataset using llama_index
        with st.expander("See dataset definition"):
            st.write(exp_output)

        # placeholders for interactive graphs
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            selector1_1, selector2_1, selector3_1 = st.columns(3)

            with selector1_1:
                selected_x1 = st.selectbox('Select X', categorical_vars)

            with selector2_1:
                selected_y1 = st.selectbox('Select Y', numeric_vars)

            with selector3_1:
                selected_z1 = st.selectbox('Select legends', categorical_vars)

            if selected_x1 == selected_z1:
                st.write('Select different X and legend variables')

            else:
                # Group by 'Genres' and 'Product Rating', calculate mean of 'User Score', and sort by mean values
                grouped_df = df.groupby([selected_x1, selected_z1]).agg({selected_y1: 'mean'}).reset_index()
                sorted_df = grouped_df.sort_values(by=selected_y1, ascending=False)

                # Select top 10 'Genres' values
                top_10_x = sorted_df[selected_x1].head(10).tolist()

                # Filter the DataFrame to include only the top 10 'Genres' values
                filtered_df = grouped_df[grouped_df[selected_x1].isin(top_10_x)]

                time.sleep(2)
                # Plot using Plotly Express
                fig = px.bar(filtered_df, x=selected_x1, y=selected_y1, color=selected_z1,
                             title='Plot 1',
                             labels={'x': 'x', 'y': 'Mean of User Score', 'z': 'Z'},
                             color_discrete_sequence=px.colors.qualitative.Set1)

                # Show the plot
                st.plotly_chart(fig)

        st.markdown("---")

        # second viz

        with fig_col2:
            selector1_2, selector2_2, selector3_2 = st.columns(3)

            with selector1_2:
                selected_x2 = st.selectbox('Select X', categorical_vars, key='a1')

            with selector2_2:
                selected_y2 = st.selectbox('Select Y', numeric_vars, key='b1')

            with selector3_2:
                selected_z2 = st.selectbox('Select legends', categorical_vars, key='c1')

            if selected_x2 == selected_z2:
                st.write('Select different X and legend variables')

            else:
                # Group by 'Genres' and 'Product Rating', calculate mean of 'User Score', and sort by mean values
                grouped_df = df.groupby([selected_x2, selected_z2]).agg({selected_y2: 'mean'}).reset_index()
                sorted_df = grouped_df.sort_values(by=selected_y2, ascending=False)

                # Select top 10 'Genres' values
                top_10_x = sorted_df[selected_x2].head(10).tolist()

                # Filter the DataFrame to include only the top 10 'Genres' values
                filtered_df = grouped_df[grouped_df[selected_x2].isin(top_10_x)]

                time.sleep(2)
                # Plot using Plotly Express
                fig = px.bar(filtered_df, x=selected_x2, y=selected_y2, color=selected_z2,
                             title='Plot 1',
                             labels={'x': 'x', 'y': 'Mean of User Score', 'z': 'Z'},
                             color_discrete_sequence=px.colors.qualitative.Set1)

                # Show the plot
                st.plotly_chart(fig)

        # chat-box
        query = st.text_area("Insert your query")

        query_with_context = f'dataset definition:\n {exp_result}' \
                             f'user prompt: {query}'

        if st.button("Submit Query"):
            result = agent.invoke({"input": main_prompt.format(query_str=query)})

            code, text = parse_response(result['output'])

            st.write(str(text))

            if ('plt' in code) or ('px' in code):
                fig = exec(code)
                st.pyplot(fig)

            with capture_output() as captured:
                exec(code)
                output = captured.getvalue() + '\n'
            st.write(output)



if __name__ == "__main__":
    main()
