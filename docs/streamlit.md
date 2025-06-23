TITLE: Initial Streamlit Library Import
DESCRIPTION: The fundamental import statement required at the beginning of any Streamlit application file. It aliases the library as `st` for convenience.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/multipage-apps/dynamic-navigation.md#_snippet_4

LANGUAGE: python
CODE:
```
import streamlit as st
```

----------------------------------------

TITLE: Introduce Streamlit Session State: st.session_state
DESCRIPTION: Streamlit introduces `st.session_state`, a new feature that allows developers to add statefulness to their applications. This enables apps to remember user interactions and data across reruns, facilitating more complex and interactive user experiences.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/quick-references/release-notes/2021.md#_snippet_4

LANGUAGE: APIDOC
CODE:
```
st.session_state

Purpose: Allows adding statefulness to Streamlit applications.
```

----------------------------------------

TITLE: Set Streamlit App Title
DESCRIPTION: Adds a main title to the Streamlit application, displayed prominently at the top of the page.
SOURCE: https://github.com/streamlit/docs/blob/main/content/get-started/fundamentals/tutorials/create-an-app.md#_snippet_1

LANGUAGE: python
CODE:
```
st.title('Uber pickups in NYC')
```

----------------------------------------

TITLE: Iterative Display of Thread Results in Streamlit using Containers
DESCRIPTION: This snippet illustrates how to display results from custom threads in a Streamlit app as they become available, rather than waiting for all threads to finish. It achieves this by initializing Streamlit containers before starting the threads and then using a polling loop to update the containers with results once a thread completes, still avoiding direct Streamlit calls from within the threads.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/app-design/multithreading.md#_snippet_1

LANGUAGE: Python
CODE:
```
import streamlit as st
import time
from threading import Thread


class WorkerThread(Thread):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay
        self.return_value = None

    def run(self):
        start_time = time.time()
        time.sleep(self.delay)
        end_time = time.time()
        self.return_value = f"start: {start_time}, end: {end_time}"


delays = [5, 4, 3, 2, 1]
result_containers = []
for i, delay in enumerate(delays):
    st.header(f"Thread {i}")
    result_containers.append(st.container())

threads = [WorkerThread(delay) for delay in delays]
for thread in threads:
    thread.start()
thread_lives = [True] * len(threads)

while any(thread_lives):
    for i, thread in enumerate(threads):
        if thread_lives[i] and not thread.is_alive():
            result_containers[i].write(thread.return_value)
            thread_lives[i] = False
    time.sleep(0.5)

for thread in threads:
    thread.join()

st.button("Rerun")
```

----------------------------------------

TITLE: Retain Streamlit Widget Value by Interrupting Cleanup
DESCRIPTION: This Python snippet demonstrates a method to prevent Streamlit from deleting a widget's data when it's not rendered on the current page. By re-saving a key-value pair in `st.session_state` at the top of every page (or in the entrypoint file for `st.navigation` apps), the widget's value is retained, ensuring it remains stateful even when navigating away and back.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/multipage-apps/widgets.md#_snippet_3

LANGUAGE: python
CODE:
```
if "my_key" in st.session_state:
    st.session_state.my_key = st.session_state.my_key
```

----------------------------------------

TITLE: API Reference for st.write function
DESCRIPTION: Documents the `st.write` function in Streamlit, which allows developers to write diverse data types, including text, dataframes, plots, and more, directly to the application interface. It serves as a primary method for displaying content.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/api-reference/write-magic/write.md#_snippet_0

LANGUAGE: APIDOC
CODE:
```
st.write(*args)
  Description: Writes arguments to the Streamlit app.
  Parameters:
    *args: Any data type or object to display in the app. This can include strings, numbers, dataframes, plots, images, Markdown, etc.
  Returns: None
```

----------------------------------------

TITLE: Illustrating Data Corruption with st.cache_resource and In-Place DataFrame Mutation
DESCRIPTION: This Python example demonstrates how `st.cache_resource`, which does not copy cached objects, can lead to data corruption. It shows an in-place modification of a DataFrame that causes a `KeyError` on subsequent reruns because the original cached object is altered.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/caching.md#_snippet_34

LANGUAGE: python
CODE:
```
@st.cache_resource   # 👈 Turn off copying behavior
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data("https://raw.githubusercontent.com/plotly/datasets/master/uber-rides-data1.csv")
st.dataframe(df)

df.drop(columns=['Lat'], inplace=True)  # 👈 Mutate the dataframe inplace

st.button("Rerun")
```

----------------------------------------

TITLE: Streaming OpenAI Chat Completions in Streamlit
DESCRIPTION: This Python code snippet shows how to integrate OpenAI's chat completions API to stream responses directly into a Streamlit chat application. It makes an API call with `stream=True`, passes the chat history, and uses `st.write_stream` to display the incoming tokens, then saves the complete response to session state.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/llms/conversational-apps.md#_snippet_16

LANGUAGE: python
CODE:
```
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

----------------------------------------

TITLE: st.write - Write arguments to the app
DESCRIPTION: The `st.write` method allows you to display various types of arguments directly to your Streamlit application. It can handle strings, dataframes, Matplotlib figures, and more, automatically rendering them appropriately.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/api-reference/_index.md#_snippet_0

LANGUAGE: python
CODE:
```
st.write("Hello **world**!")
st.write(my_data_frame)
st.write(my_mpl_figure)
```

----------------------------------------

TITLE: Download Pandas DataFrame as CSV in Streamlit
DESCRIPTION: This Python code demonstrates how to read a CSV file into a Pandas DataFrame, convert the DataFrame to a CSV formatted string, and then provide a download button in a Streamlit application. It uses `st.cache_data` to optimize the DataFrame conversion and `st.download_button` to trigger the file download.
SOURCE: https://github.com/streamlit/docs/blob/main/content/kb/FAQ/how-download-pandas-dataframe-csv.md#_snippet_0

LANGUAGE: python
CODE:
```
import streamlit as st
import pandas as pd

df = pd.read_csv("dir/file.csv")

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


csv = convert_df(df)

st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)
```

----------------------------------------

TITLE: Streamlit Anti-pattern: Buttons Nested Inside Buttons
DESCRIPTION: Demonstrates an anti-pattern where a button is nested inside another button's conditional block. The inner button will never be executed or displayed because Streamlit reruns the script from top to bottom on every interaction, and the state of the outer button is reset.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_115

LANGUAGE: python
CODE:
```
import streamlit as st

if st.button('Button 1'):
    st.write('Button 1 was clicked')
    if st.button('Button 2'):
        # This will never be executed.
        st.write('Button 2 was clicked')
```

----------------------------------------

TITLE: Complete Streamlit ChatGPT-like App Implementation
DESCRIPTION: This comprehensive Python code provides the full implementation for a ChatGPT-like application using Streamlit and the OpenAI API. It includes setup for the OpenAI client, session state management for chat history and model selection, handling user input, displaying messages, and streaming AI-generated responses.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/llms/conversational-apps.md#_snippet_17

LANGUAGE: python
CODE:
```
from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

----------------------------------------

TITLE: Access current user information with st.user
DESCRIPTION: Announcing the general availability of `st.user`, a dict-like object to access information about the current user. This provides a standardized way to retrieve user-specific details within Streamlit applications.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/quick-references/release-notes/_index.md#_snippet_1

LANGUAGE: APIDOC
CODE:
```
st.user: dict-like object
  Purpose: Access information about the current user (e.g., email, name).
```

----------------------------------------

TITLE: Display Progress and Status Messages in Streamlit
DESCRIPTION: Provides examples of various Streamlit functions for displaying user feedback, including `st.spinner` for loading states, `st.progress` for progress bars, `st.status` for detailed status updates, and visual effects like `st.balloons` and `st.snow`. It also covers different types of toast and alert messages (`st.error`, `st.warning`, `st.info`, `st.success`, `st.exception`).
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/quick-references/api-cheat-sheet.md#_snippet_21

LANGUAGE: python
CODE:
```
# Show a spinner during a process
with st.spinner(text="In progress"):
    time.sleep(3)
    st.success("Done")

# Show and update progress bar
bar = st.progress(50)
time.sleep(3)
bar.progress(100)

with st.status("Authenticating...") as s:
    time.sleep(2)
    st.write("Some long response.")
    s.update(label="Response")

st.balloons()
st.snow()
st.toast("Warming up...")
st.error("Error message")
st.warning("Warning message")
st.info("Info message")
st.success("Success message")
st.exception(e)
```

----------------------------------------

TITLE: Stateful Streamlit Counter App with Session State
DESCRIPTION: This is the corrected version of the Streamlit counter app, utilizing Session State to persist the 'count' variable across reruns. The count now increments correctly each time the button is pressed, demonstrating effective state management.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/session-state.md#_snippet_5

LANGUAGE: python
CODE:
```
import streamlit as st

st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('Increment')
if increment:
    st.session_state.count += 1

st.write('Count = ', st.session_state.count)
```

----------------------------------------

TITLE: Activate Python Virtual Environment
DESCRIPTION: These commands activate the previously created virtual environment, making its Python interpreter and installed packages available in the current terminal session. Commands vary by operating system.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_3

LANGUAGE: batch
CODE:
```
.venv\Scripts\activate.bat
```

LANGUAGE: powershell
CODE:
```
.venv\Scripts\Activate.ps1
```

LANGUAGE: bash
CODE:
```
source .venv/bin/activate
```

----------------------------------------

TITLE: Complete Simple Streamlit Chatbot GUI
DESCRIPTION: This comprehensive Python code provides a fully functional, simple chatbot GUI using Streamlit. It includes a `response_generator` function to simulate streamed responses, initializes and manages chat history, accepts user input, and displays both user and assistant messages in a conversational interface.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/llms/conversational-apps.md#_snippet_11

LANGUAGE: python
CODE:
```
import streamlit as st
import random
import time


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
```

----------------------------------------

TITLE: Streamlit App with Dynamic Navigation and User Login
DESCRIPTION: Implements a Streamlit application demonstrating dynamic navigation based on user authentication. It uses `st.session_state` to manage login status and `st.navigation` to display different sets of pages (login only vs. full menu with sections) depending on whether the user is logged in. Pages are defined using `st.Page` and grouped into sections like 'Account', 'Reports', and 'Tools'.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/multipage-apps/page-and-navigation.md#_snippet_3

LANGUAGE: python
CODE:
```
import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

dashboard = st.Page(
    "reports/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True
)
bugs = st.Page("reports/bugs.py", title="Bug reports", icon=":material/bug_report:")
alerts = st.Page(
    "reports/alerts.py", title="System alerts", icon=":material/notification_important:"
)

search = st.Page("tools/search.py", title="Search", icon=":material/search:")
history = st.Page("tools/history.py", title="History", icon=":material/history:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Reports": [dashboard, bugs, alerts],
            "Tools": [search, history]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()
```

----------------------------------------

TITLE: Importing Streamlit Library in Python
DESCRIPTION: This Python snippet imports the Streamlit library, aliasing it as `st`. This is the standard first step in any Streamlit application, providing access to all of Streamlit's functionalities for building interactive web apps.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/theming/variable-fonts.md#_snippet_11

LANGUAGE: python
CODE:
```
import streamlit as st
```

----------------------------------------

TITLE: Update Streamlit Slider with Button Callback using Session State
DESCRIPTION: This Python code demonstrates how to correctly update a Streamlit slider's value when a button is clicked. It uses a callback function (`plus_one`) assigned to the `on_click` parameter of `st.button`. The key to proper updating is to directly modify `st.session_state["slider"]` (the widget's unique key) within the callback, rather than the variable `slide_val`.
SOURCE: https://github.com/streamlit/docs/blob/main/content/kb/FAQ/widget-updating-session-state.md#_snippet_0

LANGUAGE: python
CODE:
```
# the callback function for the button will add 1 to the
# slider value up to 10
def plus_one():
    if st.session_state["slider"] < 10:
        st.session_state.slider += 1
    else:
        pass
    return

# when creating the button, assign the name of your callback
# function to the on_click parameter
add_one = st.button("Add one to the slider", on_click=plus_one, key="add_one")

# create the slider
slide_val = st.slider("Pick a number", 0, 10, key="slider")
```

----------------------------------------

TITLE: Run Streamlit Application from Terminal
DESCRIPTION: This Bash command demonstrates how to start a Streamlit application named app.py from your terminal. Ensure you are in the directory containing app.py before executing the command.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/llms/chat-response-revision.md#_snippet_3

LANGUAGE: bash
CODE:
```
streamlit run app.py
```

----------------------------------------

TITLE: Validate Streamlit installation by running hello app
DESCRIPTION: Runs the Streamlit 'Hello' application to verify that Streamlit has been installed correctly and is operational. This command launches a demo app in your web browser.
SOURCE: https://github.com/streamlit/docs/blob/main/content/get-started/installation/_index.md#_snippet_1

LANGUAGE: bash
CODE:
```
streamlit hello
```

----------------------------------------

TITLE: Streamlit Counter App with Forms and Callbacks
DESCRIPTION: Demonstrates combining `st.form` with callbacks to update multiple session state variables (count and last updated time) upon form submission. The `update_counter` function is triggered by the form's submit button, ensuring all form inputs are processed together.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/session-state.md#_snippet_9

LANGUAGE: python
CODE:
```
import streamlit as st
import datetime

st.title('Counter Example')
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.last_updated = datetime.time(0,0)

def update_counter():
    st.session_state.count += st.session_state.increment_value
    st.session_state.last_updated = st.session_state.update_time

with st.form(key='my_form'):
    st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
    st.number_input('Enter a value', value=0, step=1, key='increment_value')
    submit = st.form_submit_button(label='Update', on_click=update_counter)

st.write('Current Count = ', st.session_state.count)
st.write('Last Updated = ', st.session_state.last_updated)
```

----------------------------------------

TITLE: Cache data with st.cache_data
DESCRIPTION: Use this function decorator to cache functions that return data, such as dataframe transformations, database queries, or machine learning inference results, to improve app performance.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/api-reference/caching-and-state/_index.md#_snippet_0

LANGUAGE: python
CODE:
```
@st.cache_data
def long_function(param1, param2):
  # Perform expensive computation here or
  # fetch data from the web here
  return data
```

----------------------------------------

TITLE: Removal of st.experimental_audio_input
DESCRIPTION: As part of a scheduled deprecation, `st.experimental_audio_input` has been removed. Users should transition to using `st.audio_input` for all audio input functionalities.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/quick-references/release-notes/2025.md#_snippet_7

LANGUAGE: APIDOC
CODE:
```
Deprecated: st.experimental_audio_input
Replacement: st.audio_input
  Purpose: Notifies about the removal of an experimental audio input function.
```

----------------------------------------

TITLE: Streamlit App File Structure with Custom Configuration
DESCRIPTION: This snippet shows how to include a custom configuration file ('config.toml') for a Streamlit Community Cloud app. The configuration file must be placed in a '.streamlit/' directory at the repository's root.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_1065

LANGUAGE: text
CODE:
```
your_repository/
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── your_app.py
```

----------------------------------------

TITLE: Streamlit Single-Page App Navigation Logic
DESCRIPTION: This Python code illustrates a common pattern for navigating between different 'pages' or demos within a single Streamlit application. It uses a dictionary to map descriptive names to corresponding functions and a sidebar selectbox to allow users to choose which demo to run.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_59

LANGUAGE: python
CODE:
```
page_names_to_funcs = {
    "—": intro,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
```

----------------------------------------

TITLE: Run Streamlit Application Locally
DESCRIPTION: Provides the command-line instruction `streamlit run streamlit_app.py` to execute the Streamlit application, making it accessible in a web browser for local testing and interaction.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/llms/llm-quickstart.md#_snippet_8

LANGUAGE: bash
CODE:
```
streamlit run streamlit_app.py
```

----------------------------------------

TITLE: Displaying Pandas DataFrame with st.dataframe in Streamlit
DESCRIPTION: This Python snippet demonstrates how to use `st.dataframe` to display a Pandas DataFrame in a table-like UI within a Streamlit application. It initializes a DataFrame with sample data and then renders it, utilizing `use_container_width` for responsive display.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/app-design/dataframes.md#_snippet_0

LANGUAGE: Python
CODE:
```
import streamlit as st
import pandas as pd

df = pd.DataFrame(
    [
        {"command": "st.selectbox", "rating": 4, "is_widget": true},
        {"command": "st.balloons", "rating": 5, "is_widget": false},
        {"command": "st.time_input", "rating": 3, "is_widget": true}
    ]
)

st.dataframe(df, use_container_width=true)
```

----------------------------------------

TITLE: Read and Update Streamlit Session State Values
DESCRIPTION: Illustrates how to retrieve values from `st.session_state` using `st.write` and how to update them by assignment, supporting both dictionary-like and attribute-based syntax.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/api-reference/caching-and-state/session_state.md#_snippet_1

LANGUAGE: python
CODE:
```
# Read
st.write(st.session_state.key)

# Outputs: value
```

LANGUAGE: python
CODE:
```
st.session_state.key = 'value2'     # Attribute API
st.session_state['key'] = 'value2'  # Dictionary like API
```

----------------------------------------

TITLE: Display Line Chart in Streamlit
DESCRIPTION: Uses `st.line_chart` to display a line chart from a DataFrame. This native Streamlit function is suitable for showing trends over time or continuous data.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/api-reference/charts/_index.md#_snippet_2

LANGUAGE: python
CODE:
```
st.line_chart(my_data_frame)
```

----------------------------------------

TITLE: Complete Streamlit Fragment Function for Data Streaming
DESCRIPTION: Defines the `show_latest_data` fragment function, decorated with `@st.fragment(run_every=run_every)`, enabling dynamic rerunning based on the `run_every` variable. This function retrieves the last timestamp, updates the 'data' in Session State by concatenating new data, trims the dataset to the last 100 entries, and displays it as a line chart.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/execution-flow/fragments/start-and-stop-fragment-auto-reruns.md#_snippet_18

LANGUAGE: python
CODE:
```
@st.fragment(run_every=run_every)
def show_latest_data():
    last_timestamp = st.session_state.data.index[-1]
    st.session_state.data = pd.concat(
        [st.session_state.data, get_recent_data(last_timestamp)]
    )
    st.session_state.data = st.session_state.data[-100:]
    st.line_chart(st.session_state.data)
```

----------------------------------------

TITLE: Cache ML Model Loading with st.cache_resource
DESCRIPTION: Illustrates how to use `@st.cache_resource` to cache the loading of a machine learning model, such as a Hugging Face `transformers` pipeline. This ensures the model is loaded only once globally across all users and sessions, significantly improving app startup time and reducing memory footprint by treating the model as a singleton resource.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/caching.md#_snippet_9

LANGUAGE: python
CODE:
```
from transformers import pipeline

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

query = st.text_input("Your query", value="I love Streamlit! 🎈")
if query:
    result = model(query)[0]  # 👈 Classify the query text
    st.write(result)
```

----------------------------------------

TITLE: Display DataFrame with Streamlit st.write()
DESCRIPTION: Demonstrates the versatile `st.write()` function to display a Pandas DataFrame. Streamlit automatically renders the DataFrame as an interactive table. This function is a general-purpose tool for various data types.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_12

LANGUAGE: python
CODE:
```
import streamlit as st
import pandas as pd

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
```

----------------------------------------

TITLE: Configure OpenAI API Key in Streamlit Secrets
DESCRIPTION: This snippet shows how to securely store your OpenAI API key in Streamlit's secrets management system. By creating a `.streamlit/secrets.toml` file and adding the `OPENAI_API_KEY` variable, you can access your API key safely within your Streamlit application without hardcoding it.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/llms/conversational-apps.md#_snippet_13

LANGUAGE: toml
CODE:
```
OPENAI_API_KEY = "YOUR_API_KEY"
```

----------------------------------------

TITLE: Generating and Displaying Chatbot Responses in Streamlit
DESCRIPTION: This snippet shows how to generate a simple echo response based on user input. It displays the 'assistant's' response within a chat message container using `st.markdown` and adds this response to the `st.session_state.messages` list, maintaining the chat history.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_762

LANGUAGE: python
CODE:
```
response = f"Echo: {prompt}"
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
```

----------------------------------------

TITLE: Complete Streamlit Echo Chatbot Application
DESCRIPTION: This comprehensive example demonstrates a full Streamlit echo chatbot. It initializes chat history, displays past messages, handles new user input, and generates an immediate echo response, updating the chat history accordingly.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/llms/conversational-apps.md#_snippet_7

LANGUAGE: python
CODE:
```
import streamlit as st

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
```

----------------------------------------

TITLE: Cache Machine Learning Models with `st.cache_resource` in Streamlit
DESCRIPTION: This example illustrates the use of `st.cache_resource` for caching large machine learning models. Caching models like PyTorch's ResNet50 prevents them from being reloaded into memory for every new user session, significantly improving app performance and reducing memory consumption. The model is loaded once and then reused globally.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/caching.md#_snippet_11

LANGUAGE: python
CODE:
```
@st.cache_resource
def load_model():
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()
```

----------------------------------------

TITLE: Cache Python Function in Streamlit with st.cache_data
DESCRIPTION: This snippet shows how to cache a Python function using the @st.cache_data decorator in Streamlit. It ensures that long_running_function's return value is stored and reused if called with the same parameters and code, significantly improving app performance by avoiding redundant executions. The cache automatically updates if the function's code changes during development.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/caching.md#_snippet_0

LANGUAGE: python
CODE:
```
@st.cache_data
def long_running_function(param1, param2):
    return …
```

----------------------------------------

TITLE: Create a basic Streamlit 'Hello World' app
DESCRIPTION: Create a Python file named `app.py` in your project directory. This file will contain a simple Streamlit script that writes 'Hello world' to the web interface.
SOURCE: https://github.com/streamlit/docs/blob/main/content/get-started/installation/command-line.md#_snippet_5

LANGUAGE: python
CODE:
```
import streamlit as st

st.write("Hello world")
```

----------------------------------------

TITLE: Install & Import Streamlit
DESCRIPTION: Instructions for installing Streamlit using pip and the standard import convention for Streamlit applications.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/quick-references/api-cheat-sheet.md#_snippet_0

LANGUAGE: python
CODE:
```
pip install streamlit

streamlit run first_app.py

# Import convention
>>> import streamlit as st
```

----------------------------------------

TITLE: Implement Basic Streamlit OIDC Login Flow
DESCRIPTION: This Python snippet illustrates a fundamental Streamlit application login flow. It checks if the user is logged in using st.user.is_logged_in. If not, it displays a 'Log in with Google' button which triggers st.login(). st.stop() is used to halt script execution until login, preventing further rendering. Once logged in, a 'Log out' button and a welcome message are displayed.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/connections/authentication.md#_snippet_4

LANGUAGE: python
CODE:
```
import streamlit as st

if not st.user.is_logged_in:
    if st.button("Log in with Google"):
        st.login()
    st.stop()

if st.button("Log out"):
    st.logout()
st.markdown(f"Welcome! {st.user.name}")
```

----------------------------------------

TITLE: Cache Global Resources for Performance Optimization in Streamlit
DESCRIPTION: Demonstrates the `@st.cache_resource` decorator for caching non-data objects like TensorFlow sessions or database connections. It explains how cached resources are returned by reference and how to clear specific or all cached resources, optimizing resource management.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/quick-references/api-cheat-sheet.md#_snippet_20

LANGUAGE: python
CODE:
```
# E.g. TensorFlow session, database connection, etc.
@st.cache_resource
def foo(bar):
    # Create and return a non-data object
    return session
# Executes foo
s1 = foo(ref1)
# Does not execute foo
# Returns cached item by reference, s1 == s2
s2 = foo(ref1)
# Different arg, so function foo executes
s3 = foo(ref2)
# Clear the cached value for foo(ref1)
foo.clear(ref1)
# Clear all cached entries for this function
foo.clear()
# Clear all global resources from cache
st.cache_resource.clear()
```

----------------------------------------

TITLE: Initialize Streamlit Session State Variables
DESCRIPTION: This code shows how to initialize a variable within Streamlit's Session State. It demonstrates checking for the variable's existence before assignment and illustrates both dictionary-like and attribute-based syntax for initialization.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/session-state.md#_snippet_1

LANGUAGE: python
CODE:
```
import streamlit as st

# Check if 'key' already exists in session_state
# If not, then initialize it
if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

# Session State also supports the attribute based syntax
if 'key' not in st.session_state:
    st.session_state.key = 'value'
```

----------------------------------------

TITLE: Streamlit Layout Primitives: st.columns, st.container, st.expander
DESCRIPTION: These layout primitives, previously in beta, are now stable APIs for organizing content within a Streamlit application. They allow developers to create multi-column layouts, group related elements into logical blocks, and create collapsible sections to manage screen space.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_1011

LANGUAGE: APIDOC
CODE:
```
st.columns(spec: Union[int, List[Union[int, float]]])
  Purpose: Inserts columns into the app's main area or a container.
  spec: An integer for the number of columns, or a list of numbers specifying relative widths.
st.container()
  Purpose: Inserts a multi-element container that can be used to group elements together.
  Usage: Typically used with a 'with' statement to add elements inside the container.
st.expander(label: str, expanded: bool = False)
  Purpose: Inserts a collapsible container that hides or shows its contents.
  label: A short label displayed on the expander's header.
  expanded: If True, the expander starts in an open (expanded) state.
```

----------------------------------------

TITLE: Streamlit: Installation and Basic Usage
DESCRIPTION: This snippet provides commands for installing Streamlit, running a Streamlit application from the command line, and the standard Python import convention for Streamlit.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_866

LANGUAGE: bash
CODE:
```
pip install streamlit

streamlit run first_app.py
```

LANGUAGE: python
CODE:
```
# Import convention
>>> import streamlit as st
```

----------------------------------------

TITLE: Install and Validate Streamlit via Command Line
DESCRIPTION: This snippet provides the essential command-line steps for installing Streamlit using pip and then validating the installation by running the built-in 'Hello' application. It's part of the recommended setup for local development and assumes a Python environment is already configured.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_0

LANGUAGE: Bash
CODE:
```
pip install streamlit
```

LANGUAGE: Bash
CODE:
```
streamlit hello
```

----------------------------------------

TITLE: Streamlit Login Page Unit Tests (test_app.py)
DESCRIPTION: This Python test file uses streamlit.testing.v1.AppTest to simulate user interactions and verify the behavior of the Streamlit login application. It includes tests for initial state, incorrect password attempts, successful login, and logout functionality, demonstrating how to set dummy secrets and manipulate session state for testing.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/app-testing/examples.md#_snippet_2

LANGUAGE: python
CODE:
```
from streamlit.testing.v1 import AppTest

def test_no_interaction():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.run()
    assert at.session_state["status"] == "unverified"
    assert len(at.text_input) == 1
    assert len(at.warning) == 0
    assert len(at.success) == 0
    assert len(at.button) == 0
    assert at.text_input[0].value == ""

def test_incorrect_password():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.run()
    at.text_input[0].input("balloon").run()
    assert at.session_state["status"] == "incorrect"
    assert len(at.text_input) == 1
    assert len(at.warning) == 1
    assert len(at.success) == 0
    assert len(at.button) == 0
    assert at.text_input[0].value == ""
    assert "Incorrect password" in at.warning[0].value

def test_correct_password():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.run()
    at.text_input[0].input("streamlit").run()
    assert at.session_state["status"] == "verified"
    assert len(at.text_input) == 0
    assert len(at.warning) == 0
    assert len(at.success) == 1
    assert len(at.button) == 1
    assert "Login successful" in at.success[0].value
    assert at.button[0].label == "Log out"

def test_log_out():
    at = AppTest.from_file("app.py")
    at.secrets["password"] = "streamlit"
    at.session_state["status"] = "verified"
    at.run()
    at.button[0].click().run()
    assert at.session_state["status"] == "unverified"
    assert len(at.text_input) == 1
    assert len(at.warning) == 0
    assert len(at.success) == 0
    assert len(at.button) == 0
    assert at.text_input[0].value == ""
```

----------------------------------------

TITLE: Command to Run Streamlit Application
DESCRIPTION: Instructions to start the Streamlit application from the terminal. Navigate to your project directory and execute this command.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/tutorials/multipage-apps/dynamic-navigation.md#_snippet_3

LANGUAGE: bash
CODE:
```
streamlit run streamlit_app.py
```

----------------------------------------

TITLE: Run a Streamlit app using `streamlit run`
DESCRIPTION: The easiest way to run a Streamlit app from a Python script using the `streamlit run` command, which starts a local server and opens the app in a browser.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/concepts/architecture/run-your-app.md#_snippet_0

LANGUAGE: bash
CODE:
```
streamlit run your_script.py
```

----------------------------------------

TITLE: Introduce Streamlit Multipage Apps with st.navigation and st.Page
DESCRIPTION: Streamlit introduces `st.navigation` and `st.Page` as new components for defining and managing multipage applications. These components provide a structured and preferred method for app navigation.
SOURCE: https://github.com/streamlit/docs/blob/main/content/develop/quick-references/release-notes/2024.md#_snippet_30

LANGUAGE: APIDOC
CODE:
```
st.navigation
st.Page
```

----------------------------------------

TITLE: Cache Database Queries in Streamlit
DESCRIPTION: Demonstrates using `st.cache_data` with a time-to-live (TTL) of 600 seconds to cache results from a database query. This prevents re-running the query on every app rerun, optimizing performance. It shows fetching all rows from 'mytable' and displaying them.
SOURCE: https://github.com/streamlit/docs/blob/main/public/llms-full.txt#_snippet_805

LANGUAGE: Python
CODE:
```
@st.cache_data(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

rows = run_query("SELECT * from mytable;")

# Print results.
for row in rows:
    st.write(f"{row[0]} has a :{row[1]}:")
```