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

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here")

    if uploaded_file:

        # Setting up llm agent
        df = pd.read_csv(uploaded_file)

        # initializing tools for llama_index agent
        query_agent = PandasQueryEngine(df=df, verbose=True, instruction_srt=instruction_st)

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
                ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613',
                           openai_api_key=api_key, handle_parsing_errors=True),
                df, verbose=False,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
            )

            # llama_index agent
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)

        else:

            # langchain_agent
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613'),
                df, verbose=False,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
            )

            # llama_index agent
            llm = OpenAI(model="gpt-3.5-turbo-0613", api_key=api_key)

        # llama_index agent with tools
        llm_agent = ReActAgent.from_tools(tools, llm=llm, verbose=True,
                                          context=context)

        exp_result = llm_agent.query('explain me like five each column in the dataset df in bullet points')
        exp_output = str(exp_result)

        st.title('Your Personal Data Analyst!')

        df_sum = pd.DataFrame(df.head())
        st.write("### Data Preview")
        st.write(df_sum)

        # explanation of dataset using llama_index
        with st.expander("See explanation"):
            st.write(exp_output)

        col = st.columns((8, 10, 2), gap='medium')

        with col[0]:
            st.markdown('#### Visualization 1')

            result = agent.invoke(main_prompt.format(query_str="provide python code to visualize the most insightful categorical variable and one \
            of the numeric variable from this data"))

            code, text = parse_response(result['output'])
            print(code)

            if 'plt' in code:
                fig = exec(code)
                st.pyplot(fig)

            else:

                with capture_output() as captured:
                    exec(code)
                    output = captured.getvalue() + '\n'
                st.write(output)

        with col[1]:
            st.markdown('#### Visualization 2')

            result2 = agent.invoke(main_prompt.format(query_str="provide python code to visualise one of the insightful categorical variable with maximum 10 unique categories and a corresponding insightful numeric variable"))
            print(result2)
            code, text = parse_response(result2['output'])

            print(code)

            if 'plt' in code:
                fig2 = exec(str(code))
                st.pyplot(fig2)

            else:

                with capture_output() as captured:
                    exec(code)
                    output = captured.getvalue() + '\n'
                st.write(output)


if __name__ == "__main__":
    main()