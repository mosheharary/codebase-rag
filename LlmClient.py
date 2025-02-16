import os
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
import logging

class LlmClient:
    def __init__(self, logger, model,embedding_model,openai_api_key: str):
        """
        Initialize the analyzer with API keys and setup LangChain components.
        """
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name=model,
            temperature=0,
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model=embedding_model,
        )
        
        # Initialize the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["file_name", "content"],
            template="""Analyze the provided Java file at the bottom and create a comprehensive description that includes:
                    The file name {file_name} in the first line
                    Main purpose and functionality of the code including key features and behaviors, referencing specific methods, enums and interfaces
                    Insights derived from code comments, class names, method names and variable names
                    All external Dependencies and imported class names
                    The java file:
                    {content}
            """
        )
        self.logger = logger

    def query_openai(self, query: str) -> Optional[str]:
        
        try:
            response = self.llm.invoke(query)
            return response.content
        except Exception as e:
            return f"An error occurred while using OpenAI: {str(e)}"  

    def summarize_java_file(self, file_path: str) -> str:
        """
        Reads a Java file and uses LangChain with OpenAI to generate a summary.
        
        Args:
            file_path (str): Path to the Java file
            
        Returns:
            str: Comprehensive summary of the Java file
            
        Raises:
            FileNotFoundError: If the Java file doesn't exist
            Exception: If there's an error with the API
        """
        self.logger.info(f"Get summary from OpenAI for {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"The file {file_path} was not found")
                return None
                
            # Read the Java file
            try:
                with open(file_path, 'r') as file:
                    java_content = file.read()
            except Exception as e:
                self.logger.error(f"Error reading file: {str(e)}")
                return None
                
            # Get filename and create formatted prompt
            filename = os.path.basename(file_path)
            formatted_prompt = self.prompt_template.format(
                file_name=filename,
                content=java_content
            )
            
            # Make the API call using LangChain
            try:
                # Using invoke instead of predict
                response = self.llm.invoke(formatted_prompt)
                                
                return response.content
                
            except Exception as e:
                error=str(e)
                self.logger.error(f"{error}")
                return None
        except Exception as e:
            error=str(e)
            self.logger.error(f"{error}")
            return None

    def create_embedding(self, text: str) -> Optional[list[float]]:
        """
        Generate embeddings using LangChain's OpenAI embeddings.
        
        Args:
            text (str): Text to embed
            
        Returns:
            Optional[list[float]]: The embedding vector or None if there's an error
        """
        self.logger.info(f"get embedding from OpenAI")
        
        try:
            embedding = self.embeddings.embed_query(text)
                        
            return embedding
                
        except Exception as e:
            return None
