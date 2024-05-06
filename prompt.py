from llama_index.core import PromptTemplate


instruction_st = """\
    1. Convert the query to executable Python code in Pandas or matplotlib wherever visualization is required
    2. The final line of code should be Python expression that can be called with the 'eval()' function
    3. The code should represent a solution to the query
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with pandas dataframe in Python.
    The name of the dataframe is 'df'. Fill null values first
    This is the result of 'print(df.head())':
    {df_str}
    
    Follow these instructions:
    {instruction_str}
    Query: {query_str}
    
    Expression: 
    """
)

context = """Purpose: The primary role of this agent is to act like a data analyst who is expert in pandas
    and data visualisation and answer the questions raised by the user in the query"""