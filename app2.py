import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_prompt import new_prompt
import contextlib
import sys
import io
from helper_tools import graph_plot_engine, parse_response

# Page configuration
st.set_page_config(
    page_title="Auto Analyst App",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)

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

        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613'),
            df, verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

        def_res = agent.invoke(new_prompt.format(query_str='explain me like five each column in the dataframe "df" in bullet points'))
        print(def_res)

        exp_output = str(def_res['output']).split("\n\n")[0]

        st.title('Your Personal Data Analyst!')

        df_sum = pd.DataFrame(df.head())
        st.write("### Data Preview")
        st.write(df_sum)

        with st.expander("See explanation"):
            st.write(exp_output)

        col = st.columns((8, 10, 2), gap='medium')

        with col[0]:
            st.markdown('#### Visualization 1')

            result = agent.invoke(new_prompt.format(query_str="provide python code to visualize the most insightful categorical variable and one \
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

            result2 = agent.invoke(new_prompt.format(query_str="provide python code to visualise one of the insightful categorical variable with maximum 10 unique categories and a corresponding insightful numeric variable"))
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