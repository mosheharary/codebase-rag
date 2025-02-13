from neo4j import GraphDatabase
import networkx as nx
from typing import Dict, Any, List, Set, Union
import json

class Neo4jGraphClient:
    def __init__(self, uri: str, username: str, password: str,logger):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.graph = nx.DiGraph()
        self.logger = logger

    def close(self):
        self.driver.close()


    def query_classes_and_dependencies(self, path_list: List[str], depth: int):
        with self.driver.session() as session:
            query = """
            MATCH (c:Class)
            WHERE c.path IN $path_list
            WITH c
            CALL apoc.path.subgraphNodes(c, {
                relationshipFilter: "USES>",
                maxLevel: $depth
            })
            YIELD node
            RETURN DISTINCT node.path AS class_path
            """
            result = session.run(query, path_list=path_list, depth=depth)
            return {record["class_path"] for record in result}


    def get_nodes_by_name_and_levels(self, path, levels=2):

        match_pattern = f"""
            MATCH (start {{path: $path}})
            OPTIONAL MATCH (start)-[:USES*1..{levels}]->(node)
            WHERE node:Class
            RETURN DISTINCT start.path AS startPath , node.path AS path
        """

        with self.driver.session() as session:
            result = session.run(match_pattern, path=path)
            paths = [record["path"] for record in result if record["path"] is not None]

        # Return unique paths
        unique_paths = list(set(paths))
        return unique_paths

    def get_class_usage_tree(self, class_name: str, levels: int) -> List[Dict[str, Any]]:
        """
        Retrieve a tree of Class nodes connected by USES relationships.
        Starting from a node with the given name, get all nodes that use it
        up to the specified number of levels.
        
        Args:
            class_name (str): The name of the starting Class node
            levels (int): How many levels of USES relationships to traverse
            
        Returns:
            list: List of nodes in the usage tree, or empty list if starting node not found
        """
        with self.driver.session() as session:
            # Build the query dynamically based on the number of levels
            query_parts = ["MATCH (start:Class {name: $name})"]
            
            # Add OPTIONAL MATCH clauses for each level
            for i in range(1, levels + 1):
                prev_level = "start" if i == 1 else f"level{i-1}"
                query_parts.append(f"OPTIONAL MATCH ({prev_level})<-[:USES]-(level{i}:Class)")
            
            # Build the RETURN clause with all levels
            return_levels = ["start"] + [f"level{i}" for i in range(1, levels + 1)]
            query_parts.append("RETURN DISTINCT " + ", ".join(return_levels))
            
            query = "\n".join(query_parts)
            
            result = session.run(query, {'name': class_name})
            nodes = set()  # Using a set to avoid duplicates
            
            for record in result:
                # Add each non-null node to our set
                for level in return_levels:
                    if record[level] is not None:
                        nodes.add(record[level])
                        
            return list(nodes)        


    def add_node(self, labels: Union[str, List[str]], properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new node to the graph or update if it exists.
        For Class nodes, merges the path property if it doesn't exist.
        
        Args:
            labels (Union[str, List[str]]): Node label(s)
            properties (dict): Node properties
            
        Returns:
            dict: Created/Updated node properties
        """
        with self.driver.session() as session:
            # Convert single label to list
            if isinstance(labels, str):
                labels = [labels]
                
            # Create label string for Cypher query
            label_str = ':'.join(labels)
            
            # Create the property string for MERGE
            prop_string = ', '.join(f'{k}: ${k}' for k in properties.keys())
            
            # Special handling for Class nodes with path property
            if 'Class' in labels and 'path' in properties and 'name' in properties:
                # First MERGE on name property, then SET path if it doesn't exist
                query = (
                    f"MERGE (n:Class {{name: $name}}) "
                    "ON CREATE SET n += $properties "
                    "ON MATCH SET n.path = CASE WHEN n.path IS NULL THEN $path ELSE n.path END "
                    "RETURN n"
                )
                result = session.run(query, {
                    'name': properties['name'],
                    'path': properties['path'],
                    'properties': properties
                })
            else:
                # Regular MERGE for other cases
                query = (
                    f"MERGE (n:{label_str} {{{prop_string}}})"
                    " RETURN n"
                )
                result = session.run(query, properties)
                
            record = result.single()
            return record[0] if record else None        

    def add_edge(self, source_node: Dict[str, Any], target_node: Dict[str, Any],
                relationship_type: str, properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a new edge between two nodes only if it doesn't exist.

        Args:
            source_node (dict): Source node properties for matching
            target_node (dict): Target node properties for matching
            relationship_type (str): Type of relationship
            properties (dict, optional): Edge properties for new relationships only

        Returns:
            dict: Created relationship properties or None if nodes don't exist or relationship already exists
        """
        properties = properties or {}
        with self.driver.session() as session:
            # Create WHERE clause conditions
            source_conditions = ' AND '.join(f'a.{k} = ${k}' for k in source_node.keys())
            target_conditions = ' AND '.join(f'b.{k} = ${k}_target' for k in target_node.keys())

            # Create parameters dictionary
            params = {
                **source_node,
                **{f"{k}_target": v for k, v in target_node.items()},
                'props': properties
            }

            query = (
                f"MATCH (a) WHERE {source_conditions} "
                f"MATCH (b) WHERE {target_conditions} "
                f"WITH a, b "
                f"WHERE NOT EXISTS((a)-[:{relationship_type}]->(b)) "  # Check if relationship doesn't exist
                f"CREATE (a)-[r:{relationship_type}]->(b) "
                "SET r += $props "
                "RETURN r"
            )

            try:
                result = session.run(query, params)
                record = result.single()
                return record[0] if record else None
            except Exception as e:
                print(f"Error in adding edge: {e}")
                return None
                
    def import_java_code_structure(self, json_data: Union[str, Dict], node_classes, relationship) -> None:
        """
        Import Java code structure from JSON data into the Neo4j graph.
        
        Args:
            json_data (Union[str, Dict]): JSON string or dictionary containing code structure
        """
        # If JSON string provided, parse it
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        try:
            file_path = data["file_path"]
            package = data["info"]["package"]
            # handle class usage
            for usage in data["info"]["class_usage"]:
                usage_fqcn = usage
                if usage_fqcn in node_classes:
                    fqcn_properties = {
                        "name": usage_fqcn,
                        "path": node_classes[usage_fqcn]
                    }
                #usage_node = self.add_node(["Class"], fqcn_properties)
                    for class_info in data["info"]["classes"]:
                        fqcn = f"{class_info}"
                        relationship_string = f"{fqcn}-uses-{usage_fqcn}"
                        if relationship_string not in relationship:
                            self.add_edge(
                                {"name": fqcn},
                                {"name": usage_fqcn},
                                "USES"
                            )
                            self.logger.info(f"Created Class usage relationship: ({fqcn})->({usage_fqcn})")
                            relationship.append(relationship_string)

            return relationship
        except Exception as e:
            print(f"Error importing data: {str(e)}")
            return relationship

    def clear_graph(self) -> Dict[str, int]:
        """
        Delete all nodes and relationships from the graph.
        
        Returns:
            Dict[str, int]: Dictionary containing counts of deleted nodes and relationships
        """
        with self.driver.session() as session:
            query = "MATCH (n) DETACH DELETE n"
            result = session.run(query)
                    