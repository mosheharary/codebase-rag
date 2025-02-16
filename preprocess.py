import os
import javalang
import networkx as nx
import chardet
from typing import List, Dict, Optional, Any
import hashlib
import json
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import javalang
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from Neo4jGraphClient import Neo4jGraphClient
from OpensearchClient import OpensearchClient
from PineconeClient import PineconeClient
from CodeVisitor import CodeVisitor
from LlmClient import LlmClient
import openai
import logging
import re
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_app.log"),  # Log messages to a file
        logging.StreamHandler()            # Also log messages to the console
    ]
)

logger = logging.getLogger(__name__)


class JavaKnowledgeGraphBuilder:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, logger):
        """
        Initialize the knowledge graph builder with Neo4j connection details.
        
        :param neo4j_uri: Connection URI for Neo4j database
        :param neo4j_user: Neo4j username
        :param neo4j_password: Neo4j password
        """
        self.neo4j = Neo4jGraphClient(neo4j_uri, neo4j_user, neo4j_password,logger)
        self.fname = 'graph.json'
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_MODEL")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
        self.logger = logger
        try:
            with open(self.fname, 'r') as file:
                data = json.load(file)
            self.data = data
        except:
            self.data = {}

        self.document_store = OpensearchClient(f"{os.getenv('OPENSEARCH_HOST')}:{os.getenv('OPENSEARCH_PORT')}",logger)
        self.analyzer=LlmClient(logger,self.model,self.embedding_model,self.api_key)
        self.pinecone = PineconeClient(host=f"http://{os.getenv('PINECONE_HOST')}:{os.getenv('PINECONE_PORT')}",logger=logger)
        self.data_items = {}
        self.analyze_json_file()



    def create_embedding(self, text: str):
        """
        Generate embeddings using OpenAI's API.
        
        Mental Model: Think of this as translating text into a 
        mathematical language that captures its semantic meaning.
        """
        return self.analyzer.create_embedding(text)

    def dump_graph_entris(self, dict_list):
        path = f"{self.fname}"
        destination = f"{self.fname}.bak"

        if os.path.exists(path):  
            with open(path, 'r') as source, open(destination, 'w') as dest:
                dest.write(source.read())      
            os.remove(path)
        for item in dict_list:
            self.dump_graph_entry(item)

    def test_dump_graph_entry(self, _dict):
        path = f"{self.fname}"
        
        # If file doesn't exist, create it with the first entry
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump([_dict], f)
            return True
        
        # Read the existing content
        with open(path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # If file is empty or corrupted, start a new list
                existing_data = []
        
        # Check if the dictionary already exists in the list
        if _dict not in existing_data:
            # Append the new dictionary to the existing list
            existing_data.append(_dict)
            
            # Write the updated list back to the file
            with open(path, 'w') as f:
                json.dump(existing_data, f)
            return True
        
        # Return False if the dictionary was already in the file
        return False

    def dump_graph_entry(self,_dict): 
        path = f"{self.fname}"
        with open(path, 'ab+') as f:
            f.seek(0,2)                                #Go to the end of file    
            if f.tell() == 0 :                         #Check if file is empty
                f.write(json.dumps([_dict]).encode())  #If empty, write an array
            else :
                f.seek(-1,2)           
                f.truncate()                           #Remove the last character, open the array
                f.write(' , '.encode())                #Write the separator
                f.write(json.dumps(_dict).encode())    #Dump the dictionary
                f.write(']'.encode())                  #Close the array        

    
    def parse_java_classes(self,tree) -> List[Dict]:
        """
        Parse a Java file and extract all class information, including nested classes.
        
        Args:
            file_path (str): Path to the Java file
            
        Returns:
            List[Dict]: List of dictionaries containing class information with nested classes
            named as parent.child
        """
        def process_class_node(node, parent_name: str = None) -> List[Dict]:
            """
            Recursively process a class node and its nested classes.
            
            Args:
                node: The class node to process
                parent_name: Name of the parent class if this is a nested class
                
            Returns:
                List[Dict]: List of processed class information
            """
            results = []
            
            # Construct the full class name
            class_name = f"{parent_name}.{node.name}" if parent_name else node.name
            
            class_info = {
                'name': class_name,
                'methods': [],
                'fields': [],
                'is_abstract': 'abstract' in node.modifiers,
                'modifiers': list(node.modifiers)
            }
            
            # Process methods
            for method in node.methods:
                method_info = {
                    'name': method.name,
                    'return_type': method.return_type.name if method.return_type else 'void',
                    'modifiers': list(method.modifiers),
                    'parameters': [f"{param.type.name} {param.name}" for param in method.parameters]
                }
                class_info['methods'].append(method_info)
            
            # Process fields
            for field in node.fields:
                for declarator in field.declarators:
                    field_info = {
                        'name': declarator.name,
                        'type': field.type.name,
                        'modifiers': list(field.modifiers)
                    }
                    class_info['fields'].append(field_info)
            
            results.append(class_info)
            
            # Process nested classes
            for body_declaration in node.body:
                if isinstance(body_declaration, javalang.tree.ClassDeclaration):
                    nested_results = process_class_node(body_declaration, class_name)
                    results.extend(nested_results)
                    
            return results

        try:
            classes = []
            
            # Process all top-level classes
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                # Only process top-level classes here
                if not any(isinstance(p, javalang.tree.ClassDeclaration) for p in path[:-1]):
                    classes.extend(process_class_node(node))
                
            for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
                # Only process top-level classes here
                if not any(isinstance(p, javalang.tree.InterfaceDeclaration) for p in path[:-1]):
                    classes.extend(process_class_node(node))

            for path, node in tree.filter(javalang.tree.EnumDeclaration):
                # Only process top-level classes here
                if not any(isinstance(p, javalang.tree.EnumDeclaration) for p in path[:-1]):
                    classes.extend(process_class_node(node))

            return classes
            
        except Exception as e:
            self.logger.error(f"Error parsing Java file: {e}")
            return None

    def parse_java_functions(self,tree) -> Dict[str, List[str]]:
        """
        Parse a Java file and extract all function declarations and invocations.
        
        Args:
            file_path (str): Path to the Java file
            
        Returns:
            Dict[str, List[str]]: Dictionary containing:
                - 'declarations': List of declared function names with their signatures
                - 'invocations': List of function calls found in the code
        """
        try:
            
            # Store results
            function_info = {
                'declarations': [],
                'invocations': []
            }
            
            # Extract method declarations
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                # Build method signature
                params = [f"{param.type.name} {param.name}" for param in node.parameters]
                signature = f"{node.return_type.name if node.return_type else 'void'} {node.name}({', '.join(params)})"
                function_info['declarations'].append(signature)
            
            # Extract method invocations
            for path, node in tree.filter(javalang.tree.MethodInvocation):
                # Get the method name and any qualifier (class/object name)
                qualifier = f"{node.qualifier}." if node.qualifier else ""
                method_call = f"{qualifier}{node.member}"
                function_info['invocations'].append(method_call)

            return function_info

        except javalang.parser.JavaSyntaxError as e:
            self.logger.error(f"Syntax error in Java file: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing Java file: {e}")
            return None            

    def calculate_md5(self,file_path):
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def batch_parse_java_files(self,directory: str, subdirectory: str) -> List[Dict[str, Any]]:
        """
        Batch parse Java files in a directory.
        
        :param directory: Root directory to scan for Java files
        :return: List of parsed file information
        """
        java_files = []
        err_files = []

        try:
        
        # Recursively find Java files
            for root, _, files in os.walk(directory+subdirectory):
                #for file in tqdm(files):
                for file in files:
                    if file.endswith('.java'):
                        file_path = os.path.join(root, file)
                        self.logger.info(f"[batch_parse_java_files] Processing: {file_path}")                        
                        relative_path = os.path.relpath(file_path, directory)    
                        bm25_doc, bm25_id = self.document_store.get_document_by_path(relative_path,os.getenv("OPENSEARCH_BM25_INDEX"))
                        vector_doc, vector_id = self.document_store.get_document_by_path(relative_path,os.getenv("OPENSEARCH_VECTOR_INDEX"))
                        current_md5 = self.calculate_md5(file_path)
                        if bm25_doc and vector_doc:
                            md5 = bm25_doc['metadata']['md5']
                            if current_md5 == md5:
                                summary = bm25_doc['metadata']['summary'] 
                                embedding = vector_doc['vector_field']
                                doc=vector_doc
                            else:
                                summary = None
                                embedding = None
                                md5 = None    
                                doc=None                        
                        else:
                            summary = None
                            embedding = None
                            md5 = None
                            doc=None
                         
                        
                        analyze_java_info ,err_file = self.analyze_java_file(file_path,relative_path)
                        if not analyze_java_info:
                            err_files.append(err_file)
                            self.logger.info(f"[batch_parse_java_files] Fail to parse: {file_path}")
                            continue
                        
                        if not summary:
                            summary = self.summarize_java_file(file_path)
                            if not summary:
                                self.logger.info(f"[batch_parse_java_files] Fail to summarize: {file_path}")
                                continue
                        if not embedding:
                            embedding = self.create_embedding(summary)
                            if not embedding:
                                self.logger.info(f"[batch_parse_java_files] Fail to create embedding: {file_path}")
                                continue

                        
                        self.pinecone.upsert(os.getenv("PINECONE_INDEX"),embedding,default_path=relative_path,default_summary=summary)
                        analyze_java_info['embedding'] = embedding
                        analyze_java_info['file_path'] = relative_path
                        analyze_java_info['summary'] = summary                        
                        metadata = {'summary': summary, 'md5': current_md5}
                        java_files.append(analyze_java_info)
                        if not doc:
                            self.logger.info(f"[batch_parse_java_files] Insert item to ES  {file_path}")
                            self.document_store.index_document(relative_path,metadata,embedding)
                        else:
                            self.logger.info(f"[batch_parse_java_files] item in ES , no need to index {file_path}")
                            
            #self.dump_graph_entris(java_files) 
            return java_files
        except Exception as e:
            self.dump_graph_entris(java_files) 
            return None
        
    def analyze_json_file(self):        
        for item in self.data:
            file_path = item["file_path"]
            self.data_items[file_path] = item




    def analyze_java_file(self,file_path,relative_path):
        """
        Analyze a Java file and extract key structural information.
        
        Args:
            file_path (str): Path to the Java file
        
        Returns:
            dict: Containing package, imports, classes, collections, and function details
        """
        # Validate file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
                
        # Read file content
        with open(file_path, 'rb') as rawfile:
            raw_data = rawfile.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        

        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
        except Exception as e:
        # Fallback to latin-1 if detection fails
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()            

        try:
            tree = javalang.parse.parse(content)
            #print(tree)            
        except javalang.parser.JavaSyntaxError as e:
            self.logger.error(f"Syntax error in Java file: {file_path}")            
            return None, relative_path
        
        imports= [] 
        import_map = {}
        for imp in tree.imports:
            if imp.path.startswith(f"{os.getenv('ROOT_CLASS')}"):
                imports.append(imp.path)
            if imp.wildcard:
                import_map[imp.path.split('.')[-2]] = imp.path[:-2]  # Remove .* from path
            else:
                import_map[imp.path.split('.')[-1]] = imp.path

        class_usage = []

        for class_name in import_map.keys():
            #count how many times the class_name is in content
            count = content.count(class_name)
            if count > 1:
                if import_map[class_name].startswith(f"{os.getenv('ROOT_CLASS')}"):
                    class_usage.append(import_map[class_name])        


        visitor = CodeVisitor(tree.package.name if tree.package else None, import_map)
        visitor.visit(tree)
        
        classes = self.parse_java_classes(tree)
        pkg_name = tree.package.name if tree.package else 'default'
        class_defs = []
        for cls in classes:
            class_defs.append(f"{pkg_name}.{cls['name']}")

        info = {
            'package': pkg_name,
            'imports': imports,
            'classes': class_defs,
            'class_usage': class_usage,
            'function_calls': visitor.function_calls
        }

        result = {
            'file_path': file_path,
            'info': info
        }
        return result , None        
    

    def extract_function_calls(self, tree, import_map):
        """
        Extract all function calls from the Java AST with their fully qualified class names.
        
        Args:
            tree: The Java AST
            import_map: Dictionary mapping short names to fully qualified names
        
        Returns:
            list: List of dictionaries containing function call information
        """
        function_calls = []
        
        class FunctionCallVisitor:
            def __init__(self, import_map):
                self.calls = []
                self.import_map = import_map
                self.current_package = tree.package.name if tree.package else None
            
            def resolve_fqcn(self, qualifier):
                if qualifier is None:
                    return None
                
                # If qualifier is in import map, use the fully qualified name
                if qualifier in self.import_map:
                    return f"{self.import_map[qualifier]}"
                
                # If qualifier contains dots, it might already be fully qualified
                if '.' in qualifier:
                    return qualifier
                
                # If qualifier is a single name, it might be from the current package
                if self.current_package:
                    return f"{self.current_package}.{qualifier}"
                
                return qualifier

            def visit(self, node):
                if isinstance(node, javalang.tree.MethodInvocation):
                    call_info = {
                        'name': node.member,
                        'arguments': len(node.arguments),
                        'qualifier': node.qualifier,
                        'line': node.position[0] if node.position else None
                    }
                    
                    # Resolve FQCN
                    if node.qualifier:
                        call_info['fqcn'] = self.resolve_fqcn(node.qualifier)
                    
                    self.calls.append(call_info)
                
                # Visit all children
                for child in node.children:
                    if isinstance(child, javalang.ast.Node):
                        self.visit(child)
                    elif isinstance(child, list):
                        for item in child:
                            if isinstance(item, javalang.ast.Node):
                                self.visit(item)

        visitor = FunctionCallVisitor(import_map)
        visitor.visit(tree)
        return visitor.calls    
    
    def get_dynamic_prompt(self,file_name,content):
        prompt = f"""Analyze the provided Java file at the bottom and create a comprehensive description that includes:
                    The file name {file_name} in the first line
                    Main purpose and functionality of the code including key features and behaviors, referencing specific methods, enums and interfaces
                    Insights derived from code comments, class names, method names and variable names
                    All external Dependencies and imported class names
                    The java file:
                    {content}
            """
        return prompt


    def summarize_java_file(self,file_path):
        """
        Reads a Java file and uses OpenAI to generate a 3-sentence summary.
        
        Args:
            file_path (str): Path to the Java file
            
        Returns:
            str: Three-sentence summary of the Java file
            
        Raises:
            FileNotFoundError: If the Java file doesn't exist
            Exception: If there's an error with the OpenAI API
        """        

        return self.analyzer.summarize_java_file(file_path)
            
    def analyze_interface_description(self,text):
        """
        Analyzes a text description to determine if it describes a Java interface
        and extracts the interface name if found.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            tuple: (bool, str) - (whether text describes an interface, interface name if found)
        """
        # Check if the text mentions "interface"
        contains_interface = "interface" in text.lower()
        
        # Try to find the interface name using regex
        # Look for patterns like "interface named/called <name>" or "interface <name>"
        interface_name = None
        if contains_interface:
            patterns = [
                r"interface\s+named\s+`([^`]+)`",
                r"interface\s+called\s+`([^`]+)`",
                r"interface\s+`([^`]+)`"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    interface_name = match.group(1)
                    break
        
        # Analyze if the text has typical interface characteristics
        has_interface_characteristics = all([
            "method" in text.lower(),
            "return" in text.lower(),
            contains_interface
        ])
        
        # Prepare the analysis results
        is_interface_description = contains_interface and has_interface_characteristics
        
        result = {
            "is_interface_description": is_interface_description,
            "interface_name": interface_name,
            "analysis": {
                "contains_interface_keyword": contains_interface,
                "has_interface_characteristics": has_interface_characteristics
            }
        }
        
        return result        

    def build_neo4j_graph(self,parsed_files):
        """
        Build the knowledge graph in Neo4j database.
        """
        self.neo4j.clear_graph()
        node_classes = {}
        relationship = []
        for item in parsed_files:
            file_path = item["file_path"]
            package = item["info"]["package"]
            for class_info in item["info"]["classes"]:
                fqcn = f"{class_info}"
                if fqcn not in node_classes:
                    node_classes[fqcn] = file_path
                    class_properties = {
                        "name": fqcn,
                        "path": file_path
                    }           
                    self.logger.info(f"Adding node: {fqcn}")     
                    self.neo4j.add_node(["Class"], class_properties)
                                                    
        for item in parsed_files:
            relationship = self.neo4j.import_java_code_structure(item,node_classes,relationship)
        
    def search(self, query: str, k: int = 5):
        """
        Search the knowledge graph for relevant Java classes and methods.
        
        :param query: Search query
        :param k: Number of results to return
        :return: List of search results
        """
        embedding = self.create_embedding(query)
        files_path=self.document_store.hybrid_search(query, embedding, k)
        list_of_path=[]
        for file_path in files_path:            
            list_of_path.append(file_path['metadata']['path'])
        return self.neo4j.query_classes_and_dependencies(list_of_path, k)

def main(directory: str, subdirectory: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    """
    Main function to build and visualize Java source code knowledge graph.
    
    :param directory: Directory containing Java source files
    :param neo4j_uri: Neo4j database URI
    :param neo4j_user: Neo4j username
    :param neo4j_password: Neo4j password
    """
    if subdirectory == '/':
        logger.info(f"Using root as subdirectory")
    else:
        subdirectory = f"/{subdirectory}"

    # Create knowledge graph builder
    kg_builder = JavaKnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password,logger)
    
    # Build the knowledge graph
    try:
        parsed_files = kg_builder.batch_parse_java_files(directory,subdirectory)
    except Exception as e:
        logger.error(f"Error parsing Java files: {e}")
        parsed_files = []
    if len(parsed_files) > 0:
        logger.info(f"Total files parsed: {len(parsed_files)}")
    
    # Visualize in Neo4j
        if subdirectory == "/":
            if os.getenv("PROCESS_GRAPH") == "true":
                kg_builder.build_neo4j_graph(parsed_files)
                logger.info(f"Knowledge graph built and visualized for directory: {directory}")
    else:
        return

# Example usage
if __name__ == "__main__":
    # Replace with your actual Neo4j connection details
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the main function with optional subdirectory parameter")
    parser.add_argument(
        "--subdirectory",
        type=str,
        default="/",
        help="Subdirectory parameter (default is '/')"
    )
    args = parser.parse_args()
    root_dir=os.getenv("ROOT_DIR")
    neo4j_host=os.getenv("NEO4J_HOST")
    neo4j_port=os.getenv("NEO4J_PORT")
    neo4j_pass=os.getenv("NEO4J_PASS")

    main(
        directory=f"{root_dir}",
        subdirectory=args.subdirectory,
        neo4j_uri=f"bolt://{neo4j_host}:{neo4j_port}",
        neo4j_user='neo4j',
        neo4j_password=f"{neo4j_pass}"
    )
