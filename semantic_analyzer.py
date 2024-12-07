'''
This part is the file containing the semantic analyzer, checking for semantic errors (like undeclared variables) and executing the code.
'''
import tkinter as tk

class SemanticAnalyzer:
    def __init__(self, symbol_table, ast, console):
        self.ast = ast
        self.symbol_table = symbol_table  #hold variables and functions
        self.console = console

    def run(self):
        #store functions defined in the symbol table
        for function in self.ast["functions"]:
            self.symbol_table[function["name"]] = function

        # print(self.symbol_table)
            
        #variables declared between WAZZUP and BUHBYE should already be in the symbol table (added during syntax analysis)
        variables = self.ast["main_program"]["variables"]
        for variable in variables:
            if variable["name"] not in self.symbol_table:
                if variable["value"]["type"] == "Math":
                    self.symbol_table[variable["name"]] = ("Integer Literal", self.execute_math(variable["value"]))

        statements = self.ast["main_program"]["statements"]

        #evaluate each statement
        for statement in statements:
            # print(statement)
            self.execute_statement(statement)


    def execute_statement(self, node):
        if node["type"] == "Print":
            self.console_output(node["value"])
        elif node["type"] == "Scan":
            # check if variable was declared by checking in the symbol_table
            if node["variable"] not in self.symbol_table:
                raise RuntimeError(f"Undefined variable: {node['variable']}")
            else:
                self.console_input(node["variable"])
        elif node["type"] == "Assignment":
            self.execute_assignment(node["variable"], node["value"])
            print(self.symbol_table[node["variable"]])
        elif node["type"] == "Retype":
            self.execute_retype(node["variable"], node["retyping"])
            
        # elif node["type"] == "FunctionCall":
        #     self.execute_function_call(node)

    def input_dialog(self, title, prompt):
        def on_enter():
            nonlocal user_input
            user_input = entry.get()
            dialog.destroy()

        #create the dialog window (modal)
        dialog = tk.Toplevel()
        dialog.title(title)
        dialog.grab_set() 
        tk.Label(dialog, text=prompt, font=("Arial", 12), wraplength=280).pack(pady=10)
        #entry widget
        entry = tk.Entry(dialog, font=("Arial", 12))
        entry.pack(pady=5, padx=10)
        entry.focus_set()
        #enter button
        tk.Button(dialog, text="Enter", font=("Arial", 12), command=on_enter).pack(pady=10)
        #bind enter key to the button
        dialog.bind("<Return>", lambda event: on_enter())
        #wait for the dialog to close
        user_input = None
        dialog.wait_window()
        return user_input

    def execute_retype(self, var_name, new_type):
        if new_type == "NUMBAR":
            self.symbol_table[var_name] = ("Float Literal", float(self.symbol_table[var_name][1]))
        elif new_type == "TROOF":
            if self.symbol_table[var_name][1] == 0:
                self.symbol_table[var_name] = ("Boolean Literal", "FAIL")
            else:
                self.symbol_table[var_name] = ("Boolean Literal", "WIN")

    def console_input(self, var_name):
        user_input = self.input_dialog("Input", "Enter input:")
        self.console.insert(tk.END, user_input + "\n")
        self.symbol_table[var_name] = ("String Literal", user_input)

    def console_output(self, to_print):
        #value to print is an operand (varident or literal)
        text = ""
        while to_print:
            item = to_print.pop(0)
            if item["type"] == "Operand" and item["classification"] == "Variable Identifier":
                #check if variable was declared by checking in the symbol_table
                text += str(self.get_var_value(item["value"]))
            elif item["type"] == "Operand" and item["classification"] == "String Literal":
                text += item["value"]
            elif item["type"] == "Operand" and item["classification"] == "Integer Literal":
                text += item["value"]
            elif item["type"] == "Operand" and item["classification"] == "Float Literal":
                text += item["value"]
            elif item["type"] == "Operand" and item["classification"] == "Boolean Literal":
                text += item["value"]
            else:
                #item to print is an expression
                if item["type"] == "Math":
                    text += str(self.execute_math(item))
                elif item["type"] == "Boolean" and (item["operator"] == "All" or item["operator"] == "Any"):
                    text += self.execute_boolean_special(item)
                elif item["type"] == "Boolean":
                    text += self.execute_boolean(item)
                elif item["type"] == "Concatenation":
                    text += self.execute_concat(item["value"])
                elif item["type"] == "Comparison" or item["type"] == "Relational":
                    text += self.execute_comparison_relational(item)
                

        text += "\n"
        self.console.insert(tk.END, text)

    def execute_comparison_relational(self, node):
        left = node["left"]
        if left["type"] == "Operand" and left["classification"] == "Variable Identifier":
            left = self.get_var_value(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "String Literal":
            left = int(left["value"])
        elif left["type"] == "Math":
            left = self.execute_math(left)
        elif not isinstance(left,float) and left["type"] == "Operand" and left["classification"] == "Boolean Literal":
            if left["value"] == "FAIL":
                left = 0
            else:
                left = 1
        elif left["type"] == "Operand" and left["classification"] == "Integer Literal":
            left = int(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "Float Literal":
            left = float(left["value"])

        right = node["right"]
        if right["type"] == "Operand" and right["classification"] == "Variable Identifier":
            right = self.get_var_value(right["value"])
        elif right["type"] == "Operand" and right["classification"] == "String Literal":
            right = int(right["value"])
        elif right["type"] == "Math":
            right = self.execute_math(right)
        elif not isinstance(right,float) and right["type"] == "Operand" and right["classification"] == "Boolean Literal":
            if right["value"] == "FAIL":
                right = 0
            else:
                right = 1
        elif right["type"] == "Operand" and right["classification"] == "Integer Literal":
            right = int(right["value"])
        elif right["type"] == "Operand" and right["classification"] == "Float Literal":
            right = float(right["value"])

        if node["operator"] == "Equal":
            return "WIN" if left == right else "FAIL"
        elif node["operator"] == "Not Equal":
            return "WIN" if left != right else "FAIL"
        elif node["operator"] == "Greater Than":
            return "WIN" if left > right else "FAIL"
        elif node["operator"] == "Less Than":
            return "WIN" if left < right else "FAIL"
        elif node["operator"] == "Greater Than or Equal":
            return "WIN" if left >= right else "FAIL"
        elif node["operator"] == "Less Than or Equal":
            return "WIN" if left <= right else "FAIL"

    def execute_assignment(self, var_name, value):
        if value["type"] == "Math":
            self.symbol_table[var_name] = ("Integer Literal", self.execute_math(value))
        elif value["type"] == "Concatenation":
            self.symbol_table[var_name] = ("String Literal", self.execute_concat(value["value"]))
        elif value["type"] == "Boolean":
            pass
        elif value["type"] == "Operand" and value["classification"] == "Variable Identifier":
            self.symbol_table[var_name] = self.symbol_table[value["value"]]
        elif value["type"] == "Operand" and value["classification"] == "String Literal":
            self.symbol_table[var_name] = ("String Literal", value["value"])
        elif value["type"] == "Operand" and value["classification"] == "Integer Literal":
            self.symbol_table[var_name] = ("Integer Literal", value["value"])
        elif value["type"] == "Operand" and value["classification"] == "Float Literal":
            self.symbol_table[var_name] = ("Float Literal", value["value"])
        elif value["type"] == "Operand" and value["classification"] == "Boolean Literal":
            self.symbol_table[var_name] = ("Boolean Literal", value["value"])
            self.get_var_value(var_name)
        elif value["type"] == "FunctionCall":
            self.execute_function_call(value)
        elif value["type"] == "Concatenation":
            self.symbol_table[var_name] = ("String Literal", self.execute_concat(value["value"]))
        elif value["type"] == "Typecast":
            if value["typing"] == "NUMBAR":
                self.symbol_table[var_name] = ("Float Literal", float(self.symbol_table[value["variable"]][1]))
            elif value["typing"] == "TROOF":
                actual_val = self.get_var_value(value["variable"])
                if actual_val == "0":
                    self.symbol_table[var_name] = ("Boolean Literal", "FAIL")
                else:
                    self.symbol_table[var_name] = ("Boolean Literal", "WIN")
        else:
            raise RuntimeError(f"Invalid assignment value: {value}")

    def execute_concat(self, string):
        text = ""
        while string:
            item = string.pop(0)
            if item["type"] == "Operand" and item["classification"] == "Variable Identifier":
                #check if variable was declared by checking in the symbol_table
                text += str(self.get_var_value(item["value"]))
            elif item["type"] == "Operand" and item["classification"] == "String Literal":
                text += item["value"]
        return text


    def get_var_value(self, var_name):
        if var_name not in self.symbol_table:
            raise RuntimeError(f"Undefined variable: {var_name}")
        # print(self.symbol_table[var_name][1])
        return self.symbol_table[var_name][1]
    
    def execute_boolean_special(self, node):
        if node["operator"] == "All":
            for operand in node["operands"]:
                if operand["type"] == "Operand" and operand["classification"] == "Variable Identifier":
                    if self.get_var_value(operand["value"]) == "FAIL":
                        return "FAIL"
                elif operand["type"] == "Operand" and operand["classification"] == "Boolean Literal":
                    if operand["value"] == "FAIL":
                        return "FAIL"
                elif operand["type"] == "Operand" and operand["classification"] == "Integer Literal":
                    if operand["value"] == "0":
                        return "FAIL"
                elif operand["type"] == "Boolean":
                    if self.execute_boolean(operand) == "FAIL":
                        return "FAIL"
            return "WIN"
        elif node["operator"] == "Any":
            for operand in node["operands"]:
                if operand["type"] == "Operand" and operand["classification"] == "Variable Identifier":
                    if self.get_var_value(operand["value"]) == "WIN":
                        return "WIN"
                elif operand["type"] == "Operand" and operand["classification"] == "Boolean Literal":
                    if operand["value"] == "WIN":
                        return "WIN"
                elif operand["type"] == "Operand" and operand["classification"] == "Integer Literal":
                    if operand["value"] != "0":
                        return "WIN"
                elif operand["type"] == "Boolean":
                    if self.execute_boolean(operand) == "WIN":
                        return "WIN"
            return "FAIL"
    
    def execute_boolean(self, node):
        if node["operator"] == "Not":
            if node["operand"]["type"] == "Operand" and node["operand"]["classification"] == "Variable Identifier":
                return "FAIL" if self.get_var_value(node["operand"]["value"]) == "WIN" else "WIN"

        left = node["left"]
        if left["type"] == "Operand" and left["classification"] == "Variable Identifier":
            left = self.get_var_value(left["value"])
        elif left["type"] == "Operand" and (left["classification"] == "String Literal" or left["classification"] == "Boolean Literal"):
            left = left["value"]
            

        right = node["right"]
        if right["type"] == "Operand" and right["classification"] == "Variable Identifier":
            right = self.get_var_value(right["value"])
        elif right["type"] == "Operand" and (right["classification"] == "String Literal" or right["classification"] == "Boolean Literal"):
            right = right["value"]
        
        if node["operator"] == "And":
            print(left, right)
            return "FAIL" if left == "FAIL" or right == "FAIL" else "WIN"
        elif node["operator"] == "Or":
            print(left, right)
            return "FAIL" if left == "FAIL" and right == "FAIL" else "WIN"
        elif node["operator"] == "Xor":
            print(left, right)
            return "WIN" if left != right else "FAIL"

        
    def execute_math(self, node):
        left = node["left"]
        if left["type"] == "Operand" and left["classification"] == "Variable Identifier":
            left = self.get_var_value(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "String Literal":
            left = int(left["value"])
        elif left["type"] == "Math":
            left = self.execute_math(left)
        elif not isinstance(left,float) and left["type"] == "Operand" and left["classification"] == "Boolean Literal":
            if left["value"] == "FAIL":
                left = 0
            else:
                left = 1
        elif left["type"] == "Operand" and left["classification"] == "Integer Literal":
            left = int(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "Float Literal":
            left = float(left["value"])

        right = node["right"]
        if right["type"] == "Operand" and right["classification"] == "Variable Identifier":
            right = self.get_var_value(right["value"])
        elif right["type"] == "Operand" and right["classification"] == "String Literal":
            right = int(right["value"])
        elif right["type"] == "Math":
            right = self.execute_math(right)
        elif not isinstance(right,float) and right["type"] == "Operand" and right["classification"] == "Boolean Literal":
            if right["value"] == "FAIL":
                right = 0
            else:
                right = 1
        elif right["type"] == "Operand" and right["classification"] == "Integer Literal":
            right = int(right["value"])
        elif right["type"] == "Operand" and right["classification"] == "Float Literal":
            right = float(right["value"])

        if node["operator"] == "Add":
            return int(left) + int(right)
        elif node["operator"] == "Subtract":
            return int(left) - int(right)
        elif node["operator"] == "Multiply":
            return int(left) * int(right)
        elif node["operator"] == "Divide":
            return int(left) / int(right)
        elif node["operator"] == "Modulo":
            return int(left) % int(right)
        elif node["operator"] == "Max":
            return max(int(left), int(right))
        elif node["operator"] == "Min":
            return min(int(left), int(right))

