'''
This part is the file containing the semantic analyzer, checking for semantic errors (like undeclared variables) and executing the code.
'''
import tkinter as tk

class SemanticAnalyzer:
    def __init__(self, symbol_table, ast, console, symbols_gui):
        self.ast = ast
        self.symbol_table = symbol_table  #hold variables
        self.functions = {}  #hold functions
        self.console = console
        self.symbols_gui = symbols_gui

    def run(self):
        #store functions defined in the symbol table
        for function in self.ast["functions"]:
            print(function["name"])
            self.functions[function["name"]] = function
        
        print(self.functions)
            
        #variables declared between WAZZUP and BUHBYE should already be in the symbol table (added during syntax analysis)
        variables = self.ast["main_program"]["variables"]
        for variable in variables:
            if variable["name"] not in self.symbol_table:
                if variable["value"]["type"] == "Math":
                    self.symbol_table[variable["name"]] = ("Integer Literal", self.execute_math(variable["value"]))

        statements = self.ast["main_program"]["statements"]

        #evaluate each statement
        for statement in statements:
            self.execute_statement(statement)


    def execute_statement(self, node):
        if node["type"] == "Print":
            self.console_output(node["value"], node["suppress_newline"])
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
        # elif node["type"] == "Switch-Case":
        #   self.execute_switchcase(node["it_value"], node["cases"], node["default"]) 
        elif node["type"] == "Loop":
            self.execute_loop(node)
        elif node["type"] == "Function Call":
            self.execute_function_call(node)
        
    def execute_return(self, node):
        if node["class"] == "Expression":
            if node["value"]["type"] == "Math":
                return ("Integer Literal", self.execute_math(node["value"]))
        elif node["class"] == "Variable Identifier":
            return (self.symbol_table[node["value"]])
        else:
            raise RuntimeError(f"Invalid return value: {node['value']}")

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

    def execute_loop(self, node):
        #loop as long as loop_condition is true
        if node["condition"] == "Loop While":
            update = int(self.get_var_value(node["variable"]))
            while self.execute_expression(node["loop_condition"]) == "WIN":
                for statement in node["body"]:
                    if statement["type"] == "Break":
                        break
                    else:
                        self.execute_statement(statement)
                if node["operation"] == "Increment Keyword":
                    update += 1
                    self.symbol_table[node["variable"]] = ("Integer Literal", str(update))
                elif node["operation"] == "Decrement Keyword":
                    update -= 1
                    self.symbol_table[node["variable"]] = ("Integer Literal", str(update))
        
        #loop as long as loop_condition is false
        elif node["condition"] == "Loop Until":
            update = int(self.get_var_value(node["variable"]))
            while self.execute_expression(node["loop_condition"]) == "FAIL":
                for statement in node["body"]:
                    if statement["type"] == "Break":
                        break
                    else:
                        self.execute_statement(statement)
                if node["operation"] == "Increment Keyword":
                    update += 1
                    self.symbol_table[node["variable"]] = ("Integer Literal", str(update))
                    self.update_value_gui(node["variable"], str(update))
                elif node["operation"] == "Decrement Keyword":
                    update -= 1
                    self.symbol_table[node["variable"]] = ("Integer Literal", str(update))
                    self.update_value_gui(node["variable"], str(update))

    def execute_retype(self, var_name, new_type):
        if new_type == "NUMBAR":
            self.symbol_table[var_name] = ("Float Literal", float(self.symbol_table[var_name][1]))
            self.update_value_gui(var_name, float(self.symbol_table[var_name][1]))
        elif new_type == "NUMBR":
            self.symbol_table[var_name] = ("Integer Literal", int(self.symbol_table[var_name][1]))
            self.update_value_gui(var_name, int(self.symbol_table[var_name][1]))
        elif new_type == "TROOF":
            if self.symbol_table[var_name][1] == 0:
                self.symbol_table[var_name] = ("Boolean Literal", "FAIL")
                self.update_value_gui(var_name, "FAIL")
            else:
                self.symbol_table[var_name] = ("Boolean Literal", "WIN")
                self.update_value_gui(var_name, "WIN")
        elif new_type == "YARN":
            self.symbol_table[var_name] = ("String Literal", str(self.symbol_table[var_name][1]))
            self.update_value_gui(var_name, str(self.symbol_table[var_name][1]))

    def console_input(self, var_name):
        user_input = self.input_dialog("Input", "Enter input:")
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, user_input + "\n")
        self.console.config(state=tk.DISABLED)
        self.symbol_table[var_name] = ("String Literal", user_input)
        self.update_value_gui(var_name, user_input)
    
    def update_value_gui(self, var_name, new_value):
        for var in self.symbols_gui.get_children():
            if self.symbols_gui.item(var, "values")[0] == var_name:
                self.symbols_gui.item(var, values=(var_name, new_value))
                break

    def console_output(self, to_print, suppress_newline):
        to_print_copy = to_print[:]
        #value to print is an operand (varident or literal)
        text = ""
        while to_print_copy:
            item = to_print_copy.pop(0)
            if item["type"] == "Operand" and item["classification"] == "Variable Identifier":
                #check if variable was declared by checking in the symbol_table
                text += str(self.get_var_value(item["value"]))
                print(text)
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
                text += str(self.execute_expression(item))
                
        self.console.config(state=tk.NORMAL)
        if suppress_newline:
            self.console.insert(tk.END, text)
        else:
            text += "\n"
            self.console.insert(tk.END, text)
        self.console.config(state=tk.DISABLED)

    def execute_expression(self, node):
        value = None
        if node["type"] == "Math":
            return self.execute_math(node)
        elif node["type"] == "Boolean" and (node["operator"] == "All" or node["operator"] == "Any"):
            return self.execute_boolean_special(node)
        elif node["type"] == "Boolean":
            return self.execute_boolean(node)
        elif node["type"] == "Concatenation":
            return self.execute_concat(node["value"])
        elif node["type"] == "Comparison" or node["type"] == "Relational":
            value = self.execute_comparison_relational(node)
            self.symbol_table["IT"] = ("Boolean Literal", value)
            self.update_value_gui("IT", value)
            return value

    def execute_comparison_relational(self, node):
        left = node["left"]
        if left["type"] == "Operand" and left["classification"] == "Variable Identifier":
            left = self.math_get_var_value(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "String Literal":
            left = int(left["value"])
        elif left["type"] == "Math":
            left = self.execute_math(left)
        elif not isinstance(left,float) and left["type"] == "Operand" and left["classification"] == "Boolean Literal":
            left = 0 if left["value"] == "FAIL" else 1
        elif left["type"] == "Operand" and left["classification"] == "Integer Literal":
            left = int(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "Float Literal":
            left = float(left["value"])

        right = node["right"]
        if right["type"] == "Operand" and right["classification"] == "Variable Identifier":
            right = self.math_get_var_value(right["value"])
        elif right["type"] == "Operand" and right["classification"] == "String Literal":
            right = int(right["value"])
        elif right["type"] == "Math":
            right = self.execute_math(right)
        elif not isinstance(right,float) and right["type"] == "Operand" and right["classification"] == "Boolean Literal":
            right = 0 if right["value"] == "FAIL" else 1
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
            self.update_value_gui(var_name, self.execute_math(value))
        elif value["type"] == "Concatenation":
            self.symbol_table[var_name] = ("String Literal", self.execute_concat(value["value"]))
            self.update_value_gui(var_name, self.execute_concat(value["value"]))
        elif value["type"] == "Boolean":
            pass
        elif value["type"] == "Operand" and value["classification"] == "Variable Identifier":
            self.symbol_table[var_name] = self.symbol_table[value["value"]]
            self.update_value_gui(var_name, value["value"])
        elif value["type"] == "Operand" and value["classification"] == "String Literal":
            self.symbol_table[var_name] = ("String Literal", value["value"])
            self.update_value_gui(var_name, value["value"])
        elif value["type"] == "Operand" and value["classification"] == "Integer Literal":
            self.symbol_table[var_name] = ("Integer Literal", value["value"])
            self.update_value_gui(var_name, value["value"])
        elif value["type"] == "Operand" and value["classification"] == "Float Literal":
            self.symbol_table[var_name] = ("Float Literal", value["value"])
            self.update_value_gui(var_name, value["value"])
        elif value["type"] == "Operand" and value["classification"] == "Boolean Literal":
            self.symbol_table[var_name] = ("Boolean Literal", value["value"])
            true_val = self.get_var_value(var_name)
            self.update_value_gui(var_name, true_val)
        elif value["type"] == "Function Call":
            self.execute_function_call(value)
        elif value["type"] == "Concatenation":
            self.symbol_table[var_name] = ("String Literal", self.execute_concat(value["value"]))
        elif value["type"] == "Typecast":
            if value["typing"] == "NUMBAR":
                self.symbol_table[var_name] = ("Float Literal", float(self.symbol_table[value["variable"]][1]))
                self.update_value_gui(var_name, str(float(self.symbol_table[value["variable"]][1])))
            elif value["typing"] == "NUMBR":
                self.symbol_table[var_name] = ("Integer Literal", int(self.symbol_table[value["variable"]][1]))
                self.update_value_gui(var_name, str(int(self.symbol_table[value["variable"]][1])))
            elif value["typing"] == "TROOF":
                actual_val = self.get_var_value(value["variable"])
                if actual_val == "0":
                    self.symbol_table[var_name] = ("Boolean Literal", "FAIL")
                    self.update_value_gui(var_name, "FAIL")
                else:
                    self.symbol_table[var_name] = ("Boolean Literal", "WIN")
                    self.update_value_gui(var_name, "WIN")
        else:
            raise RuntimeError(f"Invalid assignment value: {value}")
        
    def execute_function_call(self, node):
        function_name = node["name"]
        
        function = self.functions[function_name]
        if not function:
            raise RuntimeError(f"Function {function_name} is not defined")
        
        if len(function["param"]) != len(node["param"]):
            raise RuntimeError(f"Function {function_name} expects {len(function['param'])} arguments, got {len(node['param'])}")
        
        params = []
        for param in node["param"]:
            param = param["value"]
            if param["type"] == "Operand" and param["classification"] == "Variable Identifier":
                if param["value"] not in self.symbol_table:
                    raise RuntimeError(f"Undefined variable: {param['value']}")
                params.append(self.symbol_table[param["value"]])
            elif param["type"] == "Operand" and param["classification"] == "String Literal":
                params.append(("String Literal", param["value"]))
            elif param["type"] == "Operand" and param["classification"] == "Integer Literal":
                params.append(("Integer Literal", param["value"]))
            elif param["type"] == "Operand" and param["classification"] == "Float Literal":
                params.append(("Float Literal", param["value"]))
            elif param["type"] == "Operand" and param["classification"] == "Boolean Literal":
                params.append(("Boolean Literal", param["value"]))
            elif param["type"] == "Math":
                result = self.execute_math(param)
                if isinstance(result, int):
                    params.append(("Integer Literal", result))
                elif isinstance(result, float):
                    params.append(("Float Literal", result))
            elif param["type"] == "Concatenation":
                params.append(("String Literal", self.execute_concat(param["value"])))
            elif param["type"] == "Boolean":
                params.append(("Boolean Literal", self.execute_boolean(param)))
            elif param["type"] == "Comparison" or param["type"] == "Relational":
                params.append(("Boolean Literal", self.execute_comparison_relational(param)))
            else:
                raise RuntimeError(f"Invalid argument: {param}")
            
        for i, param in enumerate(function["param"]):
            self.symbol_table[param["name"]] = params[i]

        for i, statement in enumerate(function["statements"]):
            if statement["type"] == "Return":
                self.symbol_table["IT"] = self.execute_return(statement)
                break
            elif statement["type"] == "Break":
                self.symbol_table["IT"] = ("Type Literal", "NOOB")
                break
            else:
                if i == len(function["statements"]) - 1:
                    #if there are no GTFO/FOUND YR statement, the implicit IT variable will contain NOOB
                    self.symbol_table["IT"] = ("Type Literal", "NOOB")
                self.execute_statement(statement)


    def execute_concat(self, string):
        text = ""
        while string:
            item = string.pop(0)
            if item["type"] == "Operand" and item["classification"] == "Variable Identifier":
                #check if variable was declared by checking in the symbol_table
                text += str(self.get_var_value(item["value"]))
            # elif item["type"] == "Operand" and item["classification"] == "String Literal":
            #     text += item["value"]
            else:
                text +=item["value"]
        return text


    def get_var_value(self, var_name):
        if var_name not in self.symbol_table:
            raise RuntimeError(f"Undefined variable: {var_name}")
        return self.symbol_table[var_name][1]
    
    def math_get_var_value(self, var_name):
        if var_name not in self.symbol_table:
            raise RuntimeError(f"Undefined variable: {var_name}")
        var_tuple = self.symbol_table[var_name]
        if var_tuple[0] == "Integer Literal":
            return int(var_tuple[1])
        elif var_tuple[0] == "Float Literal":
            return float(var_tuple[1])
        elif var_tuple[0] == "String Literal":
            try:
                return int(var_tuple[1])
            except ValueError:
                try:
                    return float(var_tuple[1])
                except ValueError:
                    if var_tuple[1] == "WIN":
                        return 1
                    elif var_tuple[1] == "FAIL" or var_tuple[1] == "NOOB":
                        return 0
                    else:
                        raise RuntimeError(f"Invalid value for variable {var_name}")
        elif var_tuple[0] == "Boolean Literal":
            if var_tuple[1] == "WIN":
                return 1
            else:
                return 0
        elif var_tuple[0] == "Type Literal" and var_tuple[1] == "NOOB":
            return 0
        else:
            raise RuntimeError(f"Invalid value for variable {var_name}")
    
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
        elif left["type"] == "Operand" and left["classification"] == "Boolean Literal":
            left = left["value"]
        elif left["type"] == "Operand" and left["classification"] == "Integer Literal":
            left = "FAIL" if left["value"] == "0" else "WIN"
        elif left["type"] == "Operand" and left["classification"] == "Float Literal":
            left = "FAIL" if float(left["value"]) == 0 else "WIN"
        elif left["type"] == "Operand" and left["classification"] == "String Literal":
            left = "FAIL" if left["value"] == "" else "WIN"
            

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
            left = self.math_get_var_value(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "String Literal":
            try:
                left = int(left["value"])
            except ValueError:
                try:
                    left = float(left["value"])
                except ValueError:
                    if left["value"] == "WIN":
                        left = 1
                    elif left["value"] == "FAIL" or left["value"] == "NOOB":
                        left = 0
                    else:
                        raise RuntimeError(f"Invalid value for \"{left["value"]}\")")
        elif left["type"] == "Operand" and left["classification"] == "Boolean Literal":
            if left["value"] == "FAIL":
                left = 0
            else:
                left = 1
        elif left["type"] == "Operand" and left["classification"] == "Integer Literal":
            left = int(left["value"])
        elif left["type"] == "Operand" and left["classification"] == "Float Literal":
            left = float(left["value"])
        elif left["type"] == "Math":
            left = self.execute_math(left)

        right = node["right"]
        if right["type"] == "Operand" and right["classification"] == "Variable Identifier":
            right = self.math_get_var_value(right["value"])
        elif right["type"] == "Operand" and right["classification"] == "String Literal":
            try:
                right = int(right["value"])
            except ValueError:
                try:
                    right = float(right["value"])
                except ValueError:
                    if right["value"] == "WIN":
                        right = 1
                    elif right["value"] == "FAIL" or right["value"] == "NOOB":
                        right = 0
                    else:
                        raise RuntimeError(f"Invalid value for \"{right["value"]}\")")
        elif right["type"] == "Operand" and right["classification"] == "Boolean Literal":
            if right["value"] == "FAIL":
                right = 0
            else:
                right = 1
        elif right["type"] == "Operand" and right["classification"] == "Integer Literal":
            right = int(right["value"])
        elif right["type"] == "Operand" and right["classification"] == "Float Literal":
            right = float(right["value"])
        elif right["type"] == "Math":
            right = self.execute_math(right)

        if node["operator"] == "Add":
            return left + right
        elif node["operator"] == "Subtract":
            return left - right
        elif node["operator"] == "Multiply":
            return left * right
        elif node["operator"] == "Divide":
            if isinstance(left, int) and isinstance(right, int):
                return left // right
            else:
                return left / right
        elif node["operator"] == "Modulo":
            return left % right
        elif node["operator"] == "Max":
            if isinstance(left, int) and isinstance(right, int):
                return max(left, right)
            else:
                return float(max(left, right))
        elif node["operator"] == "Min":
            if isinstance(left, int) and isinstance(right, int):
                return min(left, right)
            else:
                return float(min(left, right))

