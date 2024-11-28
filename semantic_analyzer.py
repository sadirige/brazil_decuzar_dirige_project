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
            
        #variables declared between WAZZUP and BUHBYE should already be in the symbol table (added during syntax analysis)
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

    def console_input(self, var_name):
        user_input = self.input_dialog("Input", "Enter input:")
        self.symbol_table[var_name] = user_input

    def console_output(self, to_print):
        #value to print is an operand (varident or literal)
        text = ""
        while to_print:
            item = to_print.pop(0)
            if item["type"] == "Operand" and item["classification"] == "Variable Identifier":
                #check if variable was declared by checking in the symbol_table
                text += item["value"]
            elif item["type"] == "Operand" and item["classification"] == "String Literal":
                text += item["value"]
            else:
                #item to print is an expression
                if item["type"] == "Math":
                    print(item)
                    print(self.execute_math(item))
                    text += str(self.execute_math(item))
        text += "\n"
        self.console.insert(tk.END, text)


    def get_var_value(self, var_name):
        if var_name not in self.symbol_table:
            raise RuntimeError(f"Undefined variable: {var_name}")
        return self.symbol_table[var_name]

        
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

