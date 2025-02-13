class CodeVisitor:
    """
    Visitor class to analyze Java code for both function calls and class usage.
    """
    def __init__(self, current_package, import_map):
        self.current_package = current_package
        self.import_map = import_map
        self.function_calls = []
        self.class_usage = []
        # Track seen class usages to avoid duplicates
        self.seen_class_usages = set()
        
    def resolve_fqcn(self, class_name):
        """
        Resolve the fully qualified class name.
        """
        if class_name is None:
            return None
            
        # Handle array types
        base_type = class_name.replace('[]', '')
        array_suffix = class_name[len(base_type):]
            
        # Handle primitive types
        primitives = {
            'boolean', 'byte', 'char', 'double', 'float', 
            'int', 'long', 'short', 'void'
        }
        if base_type in primitives:
            return class_name
            
        # If already fully qualified
        if '.' in base_type:
            return class_name
            
        # Check import map
        if base_type in self.import_map:
            return self.import_map[base_type] + array_suffix
            
        # If in java.lang package (common classes)
        common_classes = {
            'String', 'Object', 'Exception', 'RuntimeException',
            'Throwable', 'System', 'Class', 'Integer', 'Long',
            'Boolean', 'Double', 'Float', 'Character', 'Byte',
            'Short', 'StringBuilder', 'StringBuffer'
        }
        if base_type in common_classes:
            return f"java.lang.{base_type}{array_suffix}"
            
        # Assume it's in the current package
        if self.current_package:
            return f"{self.current_package}.{base_type}{array_suffix}"
            
        return class_name

    def add_class_usage(self, class_name, usage_type, line):
        """
        Add class usage if not already seen.
        """
        fqcn = self.resolve_fqcn(class_name)
        if fqcn and (fqcn, usage_type) not in self.seen_class_usages:
            if fqcn.startswith(f"{os.getenv('ROOT_CLASS')}"):
                self.seen_class_usages.add((fqcn, usage_type))
                self.class_usage.append({
                    'class_name': class_name,
                    'fqcn': fqcn,
                    'usage_type': usage_type,
                    'line': line
                })

    def visit(self, node):
        """
        Visit a node in the AST.
        """
        #logger.info(f"Visiting node: {node}")
        # Track method invocations
        if isinstance(node, javalang.tree.MethodInvocation):
            fqcn =  self.resolve_fqcn(node.qualifier) if node.qualifier else None
            if fqcn and fqcn.startswith(f"{os.getenv('ROOT_CLASS')}"): 
                call_info = {
                    'name': node.member,
                    'arguments': len(node.arguments),
                    'qualifier': node.qualifier,
                    'line': node.position[0] if node.position else None,
                    'fqcn': fqcn
                }
                self.function_calls.append(call_info)

        # Track variable declarations
        elif isinstance(node, javalang.tree.LocalVariableDeclaration):
            for declarator in node.declarators:
                self.add_class_usage(
                    node.type.name,
                    'variable_declaration',
                    node.position[0] if node.position else None
                )

        # Track constructor invocations
        elif isinstance(node, javalang.tree.ClassCreator):
            self.add_class_usage(
                node.type.name,
                'constructor_invocation',
                node.position[0] if node.position else None
            )

        # Track field declarations
        elif isinstance(node, javalang.tree.FieldDeclaration):
            self.add_class_usage(
                node.type.name,
                'field_declaration',
                node.position[0] if node.position else None
            )

        # Track method parameters and return types
        elif isinstance(node, javalang.tree.MethodDeclaration):
            # Return type
            if node.return_type:
                self.add_class_usage(
                    node.return_type.name,
                    'return_type',
                    node.position[0] if node.position else None
                )
            # Parameters
            for param in node.parameters:
                self.add_class_usage(
                    param.type.name,
                    'parameter_type',
                    param.position[0] if param.position else None
                )

        # Track catch clauses (exception types)
        elif isinstance(node, javalang.tree.CatchClause):
            for exception_type in node.parameter.types:
                self.add_class_usage(
                    exception_type,
                    'exception_type',
                    node.position[0] if node.position else None
                )            

        # Track type parameters (generics)
        elif isinstance(node, javalang.tree.TypeParameter):
            if node.extends:
                for type_bound in node.extends:
                    self.add_class_usage(
                        type_bound.name,
                        'type_parameter_bound',
                        node.position[0] if node.position else None
                    )

        # Visit all children
        for child in node.children:
            if isinstance(child, javalang.ast.Node):
                self.visit(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, javalang.ast.Node):
                        self.visit(item)
