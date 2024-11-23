'''
This part is the file containing the semantic analyzer
'''

class SemanticAnalyzer:
    def __init__(self, ast):
        self.ast = ast
        self.symbol_table = {}  #hold variables and functions

    def run(self):
        #store functions defined before "HAI" in the symbol table
        for function in self.ast["pre_programext"]:
            self.symbol_table[function["name"]] = function

        #store functions defined after "KTHXBYE" in the symbol table
        for function in self.ast["post_programext"]:
            self.symbol_table[function["name"]] = function

        #execute main program (statements between "HAI" and "KTHXBYE")
        console_output = []
        for statement in self.ast["main_program"]["body"]:
            console_output.extend(self.execute_statement(statement))
        return console_output

    def execute_statement(self, node):
        if node["type"] == "Print":
            # print(node["value"])
            return [node["value"], "\n"]
        # elif node["type"] == "FunctionCall":
        #     self.execute_function_call(node)

    def execute_function_call(self, node):
        func_name = node["name"]
        if func_name not in self.symbol_table:
            raise RuntimeError(f"Undefined function: {func_name}")

        function = self.symbol_table[func_name]
        for statement in function["body"]:
            self.execute_statement(statement)