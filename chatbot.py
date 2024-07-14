import pandas as pd
import matplotlib.pyplot as plt
import chainlit as cl
import re
import chardet
import sys
import io
import json
import os
import seaborn
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

df = None

system_prompt = """You are a great assistant at python dataframe analysis. You will reply to the user's messages and provide the necessary information.
The user will ask you to provide the code to answer any question about the dataset.
Besides, Here are some requirements:
1: The pandas dataframe is already loaded in the variable "df".
2: Do not load the dataframe in the generated code!
2. The code has to save the figure of the visualization in an image called img.png do not do the plot.show().
3. Give the explanations along the code on how important is the visualization and what insights can we get
4. If the user asks for suggestions of analysis just provide the possible analysis without the code.
5. For any visualizations write only one block of code.
6. The available fields in the dataset "df" and their types are: {}"""

def get_dt_columns_info(df):
    # Get the column names and their value types
    column_types = df.dtypes
    # Convert the column_types Series to a list
    column_types_list = column_types.reset_index().values.tolist()
    infos = ""
    # Print the column names and their value types
    for column_name, column_type in column_types_list:
        infos += "{}({}),\n".format(column_name, column_type)
    return infos[:-1]

@cl.on_chat_start
async def start_chat():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your csv/xlsx dataset file to begin!", accept=["csv", "xlsx"], max_size_mb=100
        ).send()
    # Decode the file
    file_path = files[0].path
    global df
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    await cl.Message(
        content=f"`{files[0].name}` uploaded correctly!\n it contains {df.shape[0]} Rows and {df.shape[1]} Columns where each column type are:\n [{get_dt_columns_info(df)}]"
    ).send()

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_prompt.format(get_dt_columns_info(df))}],
    )

def extract_code(gpt_response):
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, gpt_response, re.DOTALL)
    if matches:
        return matches[-1]
    else:
        return None
    
def filter_rows(text):
    # Split the input string into individual rows
    lines = text.split('\n')
    filtered_lines = [line for line in lines if "pd.read_csv" not in line and "pd.read_excel" not in line and ".show()" not in line]
    filtered_text = '\n'.join(filtered_lines)
    
    return filtered_text

def interpret_code(gpt_response):
    if "```" in gpt_response:
        just_code = extract_code(gpt_response)
        
        if just_code.startswith("python"):
            just_code = just_code[len("python"):]
        
        just_code = filter_rows(just_code)
        print("Filtered and extracted code:\n", just_code)
        
        # Interpret the code
        print("Code to be interpreted.")
        
        # Redirect standard output to a string buffer
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            exec(just_code)
        except Exception as e:
            sys.stdout = old_stdout
            return str(e)
        
        # Restore original standard output
        sys.stdout = old_stdout
        
        # Return captured output
        return new_stdout.getvalue().strip()
    
    else:
        return False

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    # Delete img.png image if exists
    try:
        os.remove("img.png")
    except:
        pass

    elements = []

    # Add the user's message to the history
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message}) 
    
    model = ChatOllama(model="mistral")
    json_schema = {
        "title": "DataFrame",
        "description": "Information about a pandas DataFrame.",
        "type": "object",
        "properties": {
            "columns_info": {
                "title": "Columns Info",
                "description": "Information about columns in the DataFrame.",
                "type": "string",
            }
        },
        "required": ["columns_info"],
    }
    dumps = json.dumps(json_schema, indent=2)
    columns_info = get_dt_columns_info(df)
    messages = [
        HumanMessage(content=f"""You are a great assistant at python dataframe analysis. You will reply to the user's messages and provide the necessary information.
The user will ask you to provide the code to answer any question about the dataset.
Besides, Here are some requirements:
1: The pandas dataframe is already loaded in the variable "df".
2: Do not load the dataframe in the generated code!
2. The code has to save the figure of the visualization in an image called img.png do not do the plot.show().
3. Give the explanations along the code on how important is the visualization and what insights can we get
4. If the user asks for suggestions of analysis just provide the possible analysis without the code.
5. For any visualizations write only one block of code.
6. The available fields in the dataset "df" and their types are: {columns_info}"""),
        HumanMessage(content=dumps),
        HumanMessage(content="Generate Python code using the given message.")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    


    # Define a chain of interactions
    chain = prompt | model | StrOutputParser()

    # Invoke the chain to process the text input
    response = chain.invoke({"dumps":dumps})
    has_code = interpret_code(response)
    print(f"Has_code: {has_code}")

    final_message = ""
    if os.path.exists("./img.png"):
        # Read the image
        elements = [
            cl.Image(name="image1", display="inline", path="./img.png")
        ]

    if has_code:
        await cl.Message(content=has_code, elements=elements).send()
    else:
        final_message = "Request not possible at this time"
        await cl.Message(content=final_message, elements=elements).send()
