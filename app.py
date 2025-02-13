# app.py
import streamlit as st
import openai
import os
import json
import hashlib
from opensearchpy import OpenSearch
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import numpy as np
from OpensearchClient import OpensearchClient
from Neo4jGraphClient import Neo4jGraphClient
from LlmClient import LlmClient
import logging
import tiktoken
import google.generativeai as genai

DEFAULT_TOP_ES_K = 2
DEFAULT_GRAPH_SEARCH_LENGTH = 2
MAX_PARAMETER_VALUE = 5
MIN_PARAMETER_VALUE = 0

# Constants
USER_DB_FILE = "users.json"
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant specializing in Java development. 
Using the provided context of Java classes, answer questions from software developers about the implementation and flows of Skybox’s system.
• Analyze class names, function names, and comments to infer their purpose.
• Leverage your Java expertise, including knowledge of common open-source libraries present in the code.
• If the provided classes are insufficient, suggest follow-up questions to gather more details.
• Support your responses with relevant code examples from the context to enhance clarity."""

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
load_dotenv()
model=os.getenv("OPENAI_MODEL")
embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL")
api_key=os.getenv("OPENAI_API_KEY")

analyzer=LlmClient(logger,model,embedding_model,api_key)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
root_dir = os.getenv("ROOT_DIR")

# Initialize OpenSearch client
opensearch_client = OpensearchClient(f"{os.getenv('OPENSEARCH_HOST')}:{os.getenv('OPENSEARCH_PORT')}",logger)
neo4j_host=os.getenv("NEO4J_HOST")
neo4j_port=os.getenv("NEO4J_PORT")
neo4j_pass=os.getenv("NEO4J_PASS")
neo4j = Neo4jGraphClient(f"bolt://{neo4j_host}:{neo4j_port}", 'neo4j', neo4j_pass, logger)

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# User Management Functions
def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f)

def register_user(username, password):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = hash_password(password)
    save_users(users)
    return True, "Registration successful"

def authenticate_user(username, password):
    """Authenticate a user"""
    users = load_users()
    if username not in users:
        return False
    return users[username] == hash_password(password)

def get_embedding(text):
    try:
        return analyzer.create_embedding(text)
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return None

def read_files(file_paths):
    """Read and combine content from multiple files"""
    combined_content_list = []
    for path in file_paths:
        try:
            full_path = os.path.join(root_dir, path)
            with open(full_path, 'r') as file:
                combined_content_list.append(file.read())
        except Exception as e:
            st.error(f"Error reading file {path}: {str(e)}")
    return combined_content_list

def combined_content(context_list):
    return "\n\n".join(context_list)

def query_openai(query, context, system_prompt):
    """Query OpenAI with context using Langchain"""
    try:
        message =  f"{system_prompt}\n\n"
        message += f"Answer the following question:\n\n"
        message += f"Question: {query}\n\n"
        message += f"Basd on the following context:\n\n"
        message += f"Context: {context}"

        return analyzer.query(message)
    except Exception as e:
        logger.error(f"Error querying OpenAI: {e}")
        return f"An error occurred while using OpenAI: {str(e)}"
    
def query_gemini(query, context, system_prompt):
    """Query Gemini with context."""
    try:
        # Configure the API key
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Use the same env variable
        # Access the Gemini model
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL")) # Or gemini-pro if that's preferred

        message =  f"{system_prompt}\n\n"
        message += f"Answer the following question:\n\n"
        message += f"Question: {query}\n\n"
        message += f"Basd on the following context:\n\n"
        message += f"Context: {context}"

        response = model.generate_content(message)
        return response.text  # Extract the text from the response

    except Exception as e:
        logging.error(f"Error querying Gemini: {e}")
        return f"An error occurred while using Gemini: {str(e)}"    

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'list_of_files' not in st.session_state:
    st.session_state.list_of_files = [] # To track files for debug
if 'list_of_files_from_es' not in st.session_state:
    st.session_state.list_of_files_from_es = [] # To track files for debug
if 'top_es_k' not in st.session_state:
    st.session_state.top_es_k = DEFAULT_TOP_ES_K
if 'graph_search_length' not in st.session_state:
    st.session_state.graph_search_length = DEFAULT_GRAPH_SEARCH_LENGTH
if 'number_of_tokens' not in st.session_state:
    st.session_state.number_of_tokens = 0
if 'use_gemini' not in st.session_state:
    st.session_state.use_gemini = True


# Main UI
st.title("SB Assistant")

# Authentication UI
if not st.session_state.authenticated:
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

else:
    # Chat interface
    st.write(f"Welcome, {st.session_state.username}!")

    with st.sidebar:
        st.header("Search Settings")
        st.session_state.top_es_k = st.slider(
            "Number of OpenSearch results (top_es_k)",
            min_value=MIN_PARAMETER_VALUE,
            max_value=MAX_PARAMETER_VALUE,
            value=st.session_state.top_es_k,
            help="Number of documents to retrieve from OpenSearch"
        )
        st.session_state.graph_search_length = st.slider(
            "Graph search depth (graph_search_length)",
            min_value=MIN_PARAMETER_VALUE,
            max_value=MAX_PARAMETER_VALUE,
            value=st.session_state.graph_search_length,
            help="Number of levels to traverse in the graph search"
        )
        st.session_state.use_gemini = st.checkbox("Use Gemini", value=st.session_state.use_gemini)


    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Logout button
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.messages = []
        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    tab1, tab2 = st.tabs(["Chat", "Debug"])

    with tab1:
    # Get user input
        if prompt := st.chat_input("Ask your question"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Create embedding for the query
            embedding = get_embedding(prompt)

            # Get relevant documents using hybrid search
            try:
                file_paths = opensearch_client.hybrid_search(
                    query=prompt,
                    query_vector=embedding,
                    k=st.session_state.top_es_k
                )        
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}")
                file_paths = []

            list_of_files = []
            list_of_files_from_es = []
            for path in file_paths:
                list_of_files_from_es.append(path['path'])
                list_of_files.append(path['path'])
                for neo4j_path in neo4j.get_nodes_by_name_and_levels(path['path'], st.session_state.graph_search_length):
                    list_of_files.append(neo4j_path)
            

            # Read and combine content from files
            context_list = read_files(list_of_files)
            combined_content_text = combined_content(context_list)
            number_of_tokens = count_tokens(combined_content_text)
            length_message=f"""Total number of tokens in the combined content: {number_of_tokens} which is more that the LLM can handle
            Please update your search settings or rephrase your question."""
                
            st.session_state.list_of_files = list_of_files
            st.session_state.list_of_files_from_es = list_of_files_from_es

            # Get response from OpenAI
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.use_gemini:
                        gemini_resp=query_gemini(prompt, combined_content_text, st.session_state.system_prompt)
                        if gemini_resp:
                            out = f"Total number of tokens: {number_of_tokens}\n\n{gemini_resp}"
                            st.markdown(out)
                            st.session_state.messages.append({"role": "assistant", "content": gemini_resp})
                        else:
                            out = length_message
                            st.markdown(out)
                    else:
                        openui_resp = query_openai(prompt, combined_content_text,st.session_state.system_prompt)
                        if openui_resp:
                            out = f"Total number of tokens: {number_of_tokens}\n\n{openui_resp}"
                            st.markdown(out)
                            st.session_state.messages.append({"role": "assistant", "content": openui_resp})
                        else:
                            out = length_message
                            st.markdown(out)

                st.session_state.number_of_tokens = number_of_tokens

    with tab2:
        st.header("Debug Window")
        st.write("Number of tokes for that query:")
        st.write(st.session_state.number_of_tokens)
        st.write("-------------------------------------------------------")
        st.write("List of files triggered by the last chat command:")
        if st.session_state.list_of_files_from_es:
            for path in st.session_state.list_of_files_from_es:
                link=f"{os.getenv('GITLAB_BASE_URL')}/{os.getenv('GITLAB_BRANCH')}/{path}?ref_type=heads"
                st.write(link)
            st.write("-------------------------------------------------------")
        if st.session_state.list_of_files:
            for path in st.session_state.list_of_files:
                if path in st.session_state.list_of_files_from_es:
                    continue
                link=f"{os.getenv('GITLAB_BASE_URL')}/{os.getenv('GITLAB_BRANCH')}/{path}?ref_type=heads"
                st.write(link)
        else:
            st.write("No files were accessed in the last query.")
