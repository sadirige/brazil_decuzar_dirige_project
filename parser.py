'''
This part contains the syntax analyzer; parsing through the lexemes and checking if the syntax is correct.
'''
import tkinter as tk
class Parser:
    def __init__(self, tokens, console):
        self.tokens = tokens
        self.console = console
        self.current_index = 0
        self.symbol_table = {"IT": ("Implicit Variable", "NOOB")}      
        self.ast = {
            "functions": [],        #Comments, Multiline Comments, and Function Definitions before "HAI" and after "KTHXBYE" (functions are what's actually stored, comments are ignoreds)
            "main_program": None,   #Statements between "HAI" and "KTHXBYE"
        }

    def current_token(self):
        return self.tokens[self.current_index] if self.current_index < len(self.tokens) else None
    
    def next_token(self):
        return self.tokens[self.current_index + 1] if self.current_index + 1 < len(self.tokens) else None
    
    def next_next_token(self):
        return self.tokens[self.current_index + 2] if self.current_index + 2 < len(self.tokens) else None

    def parse_error(self, message):
        self.console.insert(tk.END, message)
        # return self.symbol_table
        raise SyntaxError(message)

    def expect(self, classification, lexeme=None):
        token = self.current_token()
        return token and token[1] == classification and token[0] == lexeme if lexeme else token and token[1] == classification

    def consume(self, classification, lexeme=None):
        if self.expect(classification, lexeme):
            self.current_index += 1
        else:
            token = self.current_token()
            self.parse_error(f"Syntax Error: Expected '{classification} {lexeme}' but found '{token[0]}-{token[1]}' at line {token[2]}.")

    def program(self):
        """
        <program>   ::= <programext> HAI <linebreak> 
                        <variable> <linebreak> <statement> <linebreak> 
                        KTHXBYE <programext>
        """
        #Parse pre-program (comments, multiline comments, and function definitions) before 'HAI'
        while self.current_token() and not self.expect("Code Delimiter", "HAI"):
            self.programext("HAI")

        #Parse main program (statements) between 'HAI' and 'KTHXBYE'
        if self.expect("Code Delimiter", "HAI"):
            self.consume("Code Delimiter", "HAI")
            self.consume("Linebreak")
            if self.expect("Comment"):
                self.comment()
            elif self.expect("Multiline Comment Delimiter", "OBTW"):    
                self.multcomment()
            if self.expect("Variable Delimiter", "WAZZUP"):
                self.ast["main_program"] = {"type": "Program", "variables": self.variable(), "statements": self.statement()}
            else:
                self.ast["main_program"] = {"type": "Program", "body": self.statement()}
            self.consume("Code Delimiter", "KTHXBYE")
            self.consume("Linebreak")
        else:
            self.parse_error("Syntax Error: No main program found due to missing HAI at the start of lolcode.")
            # return self.symbol_table, ("Syntax Error", "No main program found (missing 'HAI').")

        #Parse post-program (comments, multiline comments, and function definitions) after 'KTHXBYE'
        while self.current_token():
            self.programext("KTHXBYE")

        return self.symbol_table, ("Generated AST", self.ast)
    
    def programext(self, program_delimiter):
        """
        <programext> ::= <funcdef> <linebreak> | <comment> | <multcomment> | emptystr
        """
        if self.expect("Function Delimiter", "HOW IZ I"):
            self.ast["functions"].append(self.funcdef())

        elif self.expect("Comment"):
            self.comment()

        elif self.expect("Multiline Comment Delimiter", "OBTW"):
            self.multcomment()

        elif self.expect("Linebreak"):
            self.consume("Linebreak")

        elif self.expect("Multiline Comment Delimiter", "TLDR"):
            self.parse_error(f"Syntax Error: Unexpected TLDR found at line {self.current_token()[2]}, no OBTW found before it.")
            # raise SyntaxError(f"Unexpected TLDR found at line {self.current_token()[2]}, no OBTW found before it.")

        else:
            if program_delimiter == "HAI":
                self.parse_error(f"Syntax Error: Unexpected token found before 'HAI': {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            else:
                self.parse_error(f"Syntax Error: Unexpected token found after 'KTHXBYE': {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
        
    def funcdef(self):
        """
        <funcdef> ::= HOW IZ I funcident <linebreak> <funcbody> <linebreak> IF U SAY SO |
                      HOW IZ I funcident <param> <funcbody> <linebreak> IF U SAY SO
        """
        if self.expect("Function Delimiter", "HOW IZ I"):
            self.consume("Function Delimiter", "HOW IZ I")
            if self.expect("Variable Identifier"):
                funcident = self.current_token()[0]
                self.consume("Variable Identifier")

                param = []
                if self.expect("Linebreak"):
                    self.consume("Linebreak")
                    statements = self.funcbody()
                    self.consume("Function Delimiter", "IF U SAY SO")
                    return {"type": "Function Definition", "name": funcident, "param": param, "statements": statements}
                elif self.expect("Function Delimiter", "IF U SAY SO"):
                    self.consume("Function Delimiter", "IF U SAY SO")
                    return {"type": "Function Definition", "name": funcident, "param": param, "statements": []}
                elif self.expect("Variable Call", "YR"):
                    param.extend(self.param())
                    self.consume("Linebreak")
                    statements = self.funcbody()
                    self.consume("Function Delimiter", "IF U SAY SO")
                    return {"type": "Function Definition", "name": funcident, "param": param, "statements": statements}
                else:
                    self.parse_error(f"Syntax Error: Expected a linebreak or 'IF U SAY SO' after function definition, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            else:
                self.parse_error(f"Syntax Error: Expected a function identifier after 'HOW IZ I', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")

    def funcbody(self):
        """
        <funcbody> ::= <statement> | <statement> <funcbody> | <funcret>
        """
        statements = []
        while self.current_token() and not self.expect("Function Delimiter", "IF U SAY SO"):
            if self.current_token()[1] == "Code Delimiter":
                self.parse_error(f"Syntax Error: Unexpected code delimiter: {self.current_token()} inside function body.")
            else:
                expr_ops = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
                if self.expect("Output Keyword"):  #VISIBLE keyword (print)
                    statements.append(self.print())
                elif self.expect("Input Keyword"):  #GIMMEH keyword (scan)
                    statements.append(self.scan())
                elif self.expect("Variable Identifier") and self.next_token()[1] == "Variable Reassignment":  #(assignment)
                    statements.append(self.assignment())
                elif self.current_token()[1] in expr_ops:  #(expr)
                    statements.append(self.expr())
                elif self.expect("Concatenation"):  #SMOOSH keyword (concatenate)
                    statements.append(self.concatenate())
                elif self.expect("Typecasting Declaration", "MAEK"):  #MAEK keyword (typecast)
                    statements.append(self.typecast())
                elif self.expect("Variable Identifier") and self.next_token()[1] == "Typecasting Reassignment":  #IS NOW A keyword (retype)
                    statements.append(self.retype())
                elif self.expect("Function Call Delimiter", "I IZ"):  #I IZ keyword (funccall)
                    statements.append(self.funccall())
                elif self.expect("Loop Delimiter", "IM IN YR"):  #IM IN YR keyword (loop)
                    statements.append(self.loop())
                elif self.expect("If-Then Delimiter", "O RLY?"):  #O RLY keyword (ifthen)
                    statements.append(self.ifthen())
                elif self.expect("Switch-Case Delimiter", "WTF?"):  #WTF? keyword (switchcase)
                    statements.append(self.switchcase())
                elif self.expect("Comment"):  #BTW keyword (comment)
                    self.comment()
                elif self.expect("Multiline Comment Delimiter", "OBTW"):  #OBTW keyword (multiline comment)
                    self.multcomment()
                elif self.expect("Linebreak"):
                    self.consume("Linebreak")
                    if self.expect("Loop Delimiter", "IM OUTTA YR"):
                        break
                elif self.expect("Else If Keyword", "MEBBE"):
                    return statements
                elif self.expect("Else Keyword", "NO WAI"):
                    return statements
                elif self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                    return statements
                elif self.expect("Multiline Comment Delimiter", "TLDR"):
                    self.parse_error(f"Syntax Error: Unexpected TLDR found at line {self.current_token()[2]}, no OBTW found before it.")
                elif self.expect("Return With Value") or self.expect("Break/Return"):
                    statements.append(self.funcret())
                    break
                    #anything after GTFO or FOUND YR is ignored
                else:
                    self.parse_error(f"Syntax Error: Unexpected token in statement: {self.current_token()}, with next token {self.next_token()}")
                
        if self.current_token() is None:
            self.parse_error("Syntax Error: No IF U SAY SO found, missing function delimiter.")
        else:
            #IF U SAY SO should be next
            return statements
    
    def param(self):
        """
        <param> ::= YR varident | YR varident AN <param>
        """
        param = []
        if self.current_token()[1] == "Variable Call" and self.next_next_token()[1] == "Another One Keyword":
            self.consume("Variable Call", "YR")
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            param.append({"type": "Parameter", "name": varident})
            self.consume("Another One Keyword", "AN")
            param.extend(self.param())
        elif self.current_token()[1] == "Variable Call" and self.next_token()[1] == "Variable Identifier":
            self.consume("Variable Call", "YR")
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            param.append({"type": "Parameter", "name": varident})

        return param
    
    def param_funccall(self):
        """
        <param_funccall> ::= YR <operand> | YR <expr> | YR <operand> AN <param> | YR <expr> AN <param>
        """
        param = []
        literal_varident = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]

        self.consume("Variable Call", "YR")
        while self.current_token() and not self.expect("Function Call Delimiter", "MKAY"):
            if self.current_token()[1] in literal_varident:
                operand = self.operand()
                param.append({"type": "Parameter", "value": operand, "class": "Operand"})
            elif self.current_token()[1] in expression:
                expr = self.expr()
                param.append({"type": "Parameter", "value": expr, "class": "Expression"})
            elif self.expect("Another One Keyword", "AN"):
                self.consume("Another One Keyword", "AN")
                param.extend(self.param_funccall())
            else:
                self.parse_error(f"Syntax Error: Unexpected token found {self.current_token()}")

        if self.current_token() is None:
            self.parse_error("Syntax Error: No MKAY found, missing function call delimiter.")
        else:
            return param

    def funcret(self):
        """
        <funcret> ::= FOUND YR varident | FOUND YR <expr> | GTFO
        """
        if self.expect("Return With Value"):
            self.consume("Return With Value")
            expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
            if self.expect("Variable Identifier"):
                varident = self.current_token()[0]
                self.consume("Variable Identifier")
                self.consume("Linebreak")
                return {"type": "Return", "value": varident, "class": "Variable Identifier"}
            elif self.current_token()[1] in expression:
                expr = self.expr()
                self.consume("Linebreak")
                return {"type": "Return", "value": expr, "class": "Expression"}
            else:
                self.parse_error(f"Syntax Error: Expected a variable identifier or expression after 'FOUND YR', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                # raise SyntaxError(f"Expected a variable identifier or expression after 'FOUND YR', found {self.current_token()}")
        elif self.expect("Break/Return"):
            self.consume("Break/Return")
            self.consume("Linebreak")
            return {"type": "Break"}
        else:
            self.parse_error(f"Syntax Error: Expected 'FOUND YR' or 'GTFO', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            # raise SyntaxError(f"Expected 'FOUND YR' or 'GTFO', found {self.current_token()}")
        
    def funccall(self):
        """
        <funccall> ::= I IZ funcident MKAY | 
                       I IZ funcident <param> MKAY
        """
        if self.expect("Function Call Delimiter", "I IZ"):
            self.consume("Function Call Delimiter", "I IZ")
            if self.expect("Variable Identifier"):
                funcident = self.current_token()[0]
                self.consume("Variable Identifier")
                if self.expect("Function Call Delimiter", "MKAY"):
                    self.consume("Function Call Delimiter", "MKAY")
                    return {"type": "Function Call", "name": funcident}
                elif self.expect("Variable Call", "YR"):
                    param = self.param_funccall()
                    self.consume("Function Call Delimiter", "MKAY")
                    return {"type": "Function Call", "name": funcident, "param": param}
                else:
                    self.parse_error(f"Syntax Error: Expected 'MKAY' after function identifier, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                    # raise SyntaxError(f"Expected 'MKAY' or 'YR' after function identifier, found {self.current_token()}")
            else:
                self.parse_error(f"Syntax Error: Expected a function identifier after 'I IZ', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                # raise SyntaxError(f"Expected a function identifier after 'I IZ', found {self.current_token()}")
        else:
            self.parse_error(f"Syntax Error: Expected 'I IZ' after 'HOW IZ I', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            # raise SyntaxError(f"Expected 'I IZ' after 'HOW IZ I', found {self.current_token()}")

    def typecast(self):
        """
        <typecast> ::= MAEK varident <type>
        <type> ::= A TROOF | A NUMBR | A NUMBAR | YARN
        """
        self.consume("Typecasting Declaration")
        # self.consume("Typecasting Assignment")
        if self.expect("Variable Identifier"):
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            self.consume("Typecasting Assignment")
            if self.expect("Type Literal"):
                type = self.current_token()[0]
                self.consume("Type Literal")
                return {"type": "Typecast", "variable": varident, "typing": type}
            else:
                self.parse_error(f"Syntax Error: Expected a type after 'MAEK', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                # raise SyntaxError(f"Expected a type after 'MAEK', found {self.current_token()}")
        else:
            self.parse_error(f"Syntax Error: Expected a variable identifier after 'MAEK', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            # raise SyntaxError(f"Expected a variable identifier after 'MAEK', found {self.current_token}")

    def retype(self):
        """
        <retype> ::= varident IS NOW A <type> | varident R MAEK varident A <type>
        """
        if self.expect("Variable Identifier"):
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            if self.expect("Typecasting Reassignment"):
                self.consume("Typecasting Reassignment")
                if self.expect("Type Literal"):
                    type = self.current_token()[0]
                    self.consume("Type Literal")
                    return {"type": "Retype", "variable": varident, "retyping": type}
                else:
                    self.parse_error(f"Syntax Error: Expected a type after 'IS NOW', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                    # raise SyntaxError(f"Expected a type after 'IS NOW', found {self.current_token()}")
            elif self.expect("Typecast", "MAEK"):
                self.consume("Typecast", "MAEK")
                if self.expect("Variable Identifier"):
                    varident = self.current_token()[0]
                    self.consume("Variable Identifier")
                    if self.expect("Typecast", "A"):
                        self.consume("Typecast", "A")
                        if self.expect("Type"):
                            type = self.current_token()[0]
                            self.consume("Type")
                            return {"type": "Retype", "variable": varident, "retyping": type}
                        else:
                            self.parse_error
                            # raise SyntaxError(f"Expected a type after 'A', found {self.current_token()}")
                    else:
                        self.parse_error(f"Syntax Error: Expected 'A' after variable identifier, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                        # raise SyntaxError(f"Expected 'A' after variable identifier, found {self.current_token()}")
                else:
                    self.parse_error(f"Syntax Error: Expected a variable identifier after 'MAEK', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                    # raise SyntaxError(f"Expected a variable identifier after 'MAEK', found {self.current_token()}")
            else:
                self.parse_error(f"Syntax Error: Expected 'IS NOW' or 'MAEK' after variable identifier, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                # raise SyntaxError(f"Expected 'IS NOW' or 'MAEK' after variable identifier, found {self.current_token()}")
        else:
            self.parse_error(f"Syntax Error: Expected a variable identifier after 'MAEK', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            # raise SyntaxError(f"Expected a variable identifier after 'MAEK', found {self.current_token()}")
        
    def loop(self):
        """
        <loop> ::= IM IN YR loopident <loopop> YR varident <linebreak> <loopbody> <linebreak> IM OUTTA YR loopident | 
                   IM IN YR loopident <loopop> YR varident <loopcond> <expr> <linebreak> <loopbody> <linebreak> IM OUTTA YR loopident
        <loopop> ::= UPPIN | NERFIN
        <loopcond> ::= TIL | WILE
        """
        if self.expect("Loop Delimiter", "IM IN YR"):
            self.consume("Loop Delimiter", "IM IN YR")
            if self.expect("Variable Identifier"):
                loopident = self.current_token()[0]
                self.consume("Variable Identifier")
                if self.expect("Increment Keyword") or self.expect("Decrement Keyword"):
                    loopop = self.current_token()[1]
                    self.consume(loopop)
                    self.consume("Variable Call", "YR")
                    if self.expect("Variable Identifier"):
                        varident = self.current_token()[0]
                        self.consume("Variable Identifier")
                        if self.expect("Loop Until") or self.expect("Loop While"):
                            loopcond = self.current_token()[1]
                            self.consume(loopcond)
                            expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
                            
                            if self.current_token()[1] in expression:
                                cond = self.expr()
                                if self.expect("Linebreak"):
                                    self.consume("Linebreak")
                                    body = self.loopbody()
                                    self.consume("Loop Delimiter", "IM OUTTA YR")
                                    self.consume("Variable Identifier", loopident)
                                    return {"type": "Loop", "name": loopident, "operation": loopop, "condition": loopcond, "loop_condition": cond, "variable": varident, "body": body}
                                else:
                                    self.parse_error(f"Syntax Error: Expected a linebreak after loop body, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                            else:
                                self.parse_error(f"Syntax Error: Expected a loop condition after variable identifier, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                        else:
                            return {"type": "Loop", "name": loopident, "operation": loopop, "variable": varident, "body": self.loopbody()}
                    else:
                        self.parse_error(f"Syntax Error: Expected a variable identifier after loop operation, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                else:
                    self.parse_error(f"Syntax Error: Expected a loop operation after variable identifier, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            else:
                self.parse_error(f"Syntax Error: Expected a variable identifier after 'IM IN YR', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
        else:
            self.parse_error(f"Syntax Error: Expected 'IM IN YR' after 'HOW IZ I', found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")

    def loopbody(self):
        """
        <loopbody> ::= <statement> | <statement> <loopbody> | GTFO
        """
        statements = []
        while not self.expect("Loop Delimiter", "IM OUTTA YR"):
            if self.expect("Code Delimiter", "HAI"):
                self.parse_error(f"Syntax Error: Unexpected 'HAI' found before 'IM OUTTA YR' at line {self.current_token()[2]}.")
            elif self.current_token() is None:
                self.parse_error("Syntax Error: No 'IM OUTTA YR' found, missing loop delimiter.")
            elif self.expect("Return With Value") or self.expect("Break/Return"):
                statements.extend(self.funcret())
                break
                #anything after GTFO or FOUND YR is ignored
            else:
                statements.extend(self.statement())
        return statements
    
    def conditionals(self):
        """
        <conditionals> ::= BOTH SAEM <varident> AN <varident> | BOTH SAEM <math> AN <math> | BOTH SAEM <Integer Literal> AN <Integer Literal> |
                           BOTH SAEM <varident> AN <Integer Literal> | BOTH SAEM <varident> AN <math> |
                           DIFFRINT <varident> AN <varident> | DIFFRINT <math> AN <math> | DIFFRINT <Integer Literal> AN <Integer Literal> |
                           DIFFRINT <varident> AN <Integer Literal> | DIFFRINT <varident> AN <math>
        """
        conditionals = []   # The condition (BOTH SAEM or DIFFRINT), variable identifier (if any), math operator (BIGGR or SMALLR if any), and integer literal (if any) are stored and returned
        
        if self.expect("Equal") or self.expect("Not Equal"):
            conditionals.append(self.current_token())
            if self.expect("Equal"):
                self.consume("Equal")
                if self.expect("Variable Identifier"):
                    self.consume("Variable Identifier")
                    if self.expect("Another One Keyword"):
                        self.consume("Another One Keyword")
                        if self.expect("Variable Identifier"):
                            conditionals.append(self.current_token())
                            self.consume("Variable Identifier")
                        elif self.expect("Max") or self.expect("Min"):
                            if self.expect("Max"):
                                conditionals.append(self.math())
                            elif self.expect("Min"):
                                conditionals.append(self.math())
                        elif self.expect("Integer Literal"):
                            conditionals.append(self.current_token())
                            self.consume("Integer Literal")
                    else:
                        self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                        # raise SyntaxError(f"Expected another operator and only found one")
                elif self.expect("Max") or self.expect("Min"):
                    if self.expect("Max"):
                        conditionals.append(self.math())
                        self.consume("Max")
                        if self.expect("Another One Keyword"):
                            self.consume("Another One Keyword")
                            if self.expect("Variable Identifier"):
                                conditionals.append(self.current_token())
                                self.consume("Variable Identifier")
                            elif self.expect("Max") or self.expect("Min"):
                                if self.expect("Max"):
                                    conditionals.append(self.math())
                                elif self.expect("Min"):
                                    conditionals.append(self.math())
                            elif self.expect("Integer Literal"):
                                conditionals.append(self.current_token())
                                self.consume("Integer Literal")
                        else:
                            self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                            # raise SyntaxError(f"Expected another operator and only found one")
                    elif self.expect("Min"):
                        conditionals.append(self.math())
                        self.consume("Min")
                        if self.expect("Another One Keyword"):
                            self.consume("Another One Keyword")
                            if self.expect("Variable Identifier"):
                                conditionals.append(self.current_token())
                                self.consume("Variable Identifier")
                            elif self.expect("Max") or self.expect("Min"):
                                if self.expect("Max"):
                                    conditionals.append(self.math())
                                elif self.expect("Min"):
                                    conditionals.append(self.math())
                            elif self.expect("Integer Literal"):
                                conditionals.append(self.current_token())
                                self.consume("Integer Literal")
                        else:
                            self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                            # raise SyntaxError(f"Expected another operator and only found one")
                elif self.expect("Integer Literal"):
                    conditionals.append(self.current_token())
                    self.consume("Integer Literal")
                    if self.expect("Another One Keyword"):
                        self.consume("Another One Keyword")
                        if self.expect("Variable Identifier"):
                            conditionals.append(self.current_token())
                            self.consume("Variable Identifier")
                        elif self.expect("Max") or self.expect("Min"):
                            if self.expect("Max"):
                                conditionals.append(self.math())
                            elif self.expect("Min"):
                                conditionals.append(self.math())
                        elif self.expect("Integer Literal"):
                            conditionals.append(self.current_token())
                            self.consume("Integer Literal")
                    else:
                        self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                        # raise SyntaxError(f"Expected another operator and only found one")
            elif self.expect("Not Equal"):
                self.consume("Not Equal")
                if self.expect("Variable Identifier"):
                    self.consume("Variable Identifier")
                    if self.expect("Another One Keyword"):
                        self.consume("Another One Keyword")
                        if self.expect("Variable Identifier"):
                            conditionals.append(self.current_token())
                            self.consume("Variable Identifier")
                        elif self.expect("Max") or self.expect("Min"):
                            if self.expect("Max"):
                                conditionals.append(self.math())
                            elif self.expect("Min"):
                                conditionals.append(self.math())
                        elif self.expect("Integer Literal"):
                            conditionals.append(self.current_token())
                            self.consume("Integer Literal")
                    else:
                        self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                        # raise SyntaxError(f"Expected another operator and only found one")
                elif self.expect("Max") or self.expect("Min"):
                    if self.expect("Max"):
                        conditionals.append(self.math())
                        self.consume("Max")
                        if self.expect("Another One Keyword"):
                            self.consume("Another One Keyword")
                            if self.expect("Variable Identifier"):
                                conditionals.append(self.current_token())
                                self.consume("Variable Identifier")
                            elif self.expect("Max") or self.expect("Min"):
                                if self.expect("Max"):
                                    conditionals.append(self.math())
                                elif self.expect("Min"):
                                    conditionals.append(self.math())
                            elif self.expect("Integer Literal"):
                                conditionals.append(self.current_token())
                                self.consume("Integer Literal")
                        else:
                            self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                            # raise SyntaxError(f"Expected another operator and only found one")
                    elif self.expect("Min"):
                        conditionals.append(self.math())
                        self.consume("Min")
                        if self.expect("Another One Keyword"):
                            self.consume("Another One Keyword")
                            if self.expect("Variable Identifier"):
                                conditionals.append(self.current_token())
                                self.consume("Variable Identifier")
                            elif self.expect("Max") or self.expect("Min"):
                                if self.expect("Max"):
                                    conditionals.append(self.math())
                                elif self.expect("Min"):
                                    conditionals.append(self.math())
                            elif self.expect("Integer Literal"):
                                conditionals.append(self.current_token())
                                self.consume("Integer Literal")
                        else:
                            self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                            # raise SyntaxError(f"Expected another operator and only found one")
                elif self.expect("Integer Literal"):
                    conditionals.append(self.current_token())
                    self.consume("Integer Literal")
                    if self.expect("Another One Keyword"):
                        self.consume("Another One Keyword")
                        if self.expect("Variable Identifier"):
                            conditionals.append(self.current_token())
                            self.consume("Variable Identifier")
                        elif self.expect("Max") or self.expect("Min"):
                            if self.expect("Max"):
                                conditionals.append(self.math())
                            elif self.expect("Min"):
                                conditionals.append(self.math())
                        elif self.expect("Integer Literal"):
                            conditionals.append(self.current_token())
                            self.consume("Integer Literal")
                    else:
                        self.parse_error(f"Syntax Error: Expected another operator and only found one.")
                        # raise SyntaxError(f"Expected another operator and only found one")
        return conditionals
             

    def ifthen(self):
        """
        <ifthen> ::= <expr> <linebreak> O RLY? <linebreak> YA RLY <linebreak> <statement> <linebreak> OIC | 
                     <expr> <linebreak> O RLY? <linebreak> YA RLY <linebreak> <statement> <linebreak> NO WAI <linebreak> <statement> <linebreak> OIC
        """
        if self.expect("If-Then Delimiter", "O RLY?"):
            self.consume("If-Then Delimiter", "O RLY?")
            if self.expect("Linebreak"):
                self.consume("Linebreak")
                if self.expect("If Keyword", "YA RLY"):
                    if_body = self.if_func()
                    if self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                        self.consume("If-Then/Switch-Case Delimiter", "OIC")
                        if self.expect("Linebreak"):
                            self.consume("Linebreak")
                            return {"type": "If-Then", "if_body": if_body}
                    elif self.expect("Else If Keyword", "MEBBE"):
                        else_if_bodies = self.elif_func()
                        if self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                            self.consume("If-Then/Switch-Case Delimiter", "OIC")
                            if self.expect("Linebreak"):
                                self.consume("Linebreak")
                                return {"type": "If-Then", "condition": self.expr(), "if_body": if_body, "else_if_body": else_if_bodies}
                        elif self.expect("Else Keyword", "NO WAI"):
                            else_body = self.else_func()
                            if self.expect("Linebreak"):
                                self.consume("Linebreak")
                                if self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                                    self.consume("If-Then/Switch-Case Delimiter", "OIC")
                                    if self.expect("Linebreak"):
                                        self.consume("Linebreak")
                                        return {"type": "If-Then", "if_body": if_body, "else_if_body": else_if_bodies, "else_body": else_body}
                        else:
                            raise SyntaxError(f"Expected 'OIC' or 'NO WAI' after condition body, found {self.current_token()}")
                    elif self.expect("Else Keyword", "NO WAI"):
                        self.consume("Else Keyword", "NO WAI")
                        if self.expect("Linebreak"):
                            self.consume("Linebreak")
                            else_body = self.statement()
                            if self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                                self.consume("If-Then/Switch-Case Delimiter", "OIC")
                                if self.expect("Linebreak"):
                                    self.consume("Linebreak")
                                    return {"type": "If-Then", "if_body": if_body, "else_body": else_body}
                            else:
                                raise SyntaxError(f"Expected 'OIC' after false body, found {self.current_token()}")
                        else:
                            raise SyntaxError(f"Expected 'OIC', 'MEBBE', 'NO WAI' after condition body, found {self.current_token()}")
                else:
                    raise SyntaxError(f"Expected 'YA RLY' after linebreak, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a linebreak after 'O RLY?', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected 'O RLY?', found {self.current_token()}")
        
    def if_func(self):
        '''This should only return the body for the if-statement'''
        statement = []
        if self.expect("If Keyword", "YA RLY"):
            self.consume("If Keyword", "YA RLY")
            if self.expect("Linebreak"):
                self.consume("Linebreak")
                statement.append(self.statement())
            else:
                raise SyntaxError(f"Expected a linebreak after 'YA RLY', found {self.current_token()}")
            
        return statement
    
    def elif_func(self):
        '''This should return all the statements and their conditions in MEBBE'''
        statements = {}
        while not self.expect("Else Keyword", "NO WAI"):
            if self.expect("Else If Keyword", "MEBBE"):
                self.consume("Else If Keyword", "MEBBE")
                if self.expect("Equal") or self.expect("Not Equal") or self.expect("Max") or self.expect("Min"):
                    self.consume(self.next_token()[1])
                    condition = self.expr()
                    if self.expect("Linebreak"):
                        self.consume("Linebreak")
                        statements[condition] = self.statement()
                else:
                    raise SyntaxError(f"Expected a conditional or relational statement, found {self.current_token()}")
        
        return statements
    
    def else_func(self):
        '''This should return the statement for the else-statement'''
        statement = []
        if self.expect("Else Keyword", "NO WAI"):
            self.consume("Else Keyword", "NO WAI")
            if self.expect("Linebreak"):
                self.consume("Linebreak")
                statement.append(self.statement())
            else:
                raise SyntaxError(f"Expected a linebreak after 'NO WAI', found {self.current_token()}")
            
        return statement

    def switchcase(self):
        """
        <switchcase> ::= varident WTF? <linebreak> <case> OMGWTF <linebreak> <statement> <linebreak> OIC |
                         <literal> WTF? <linebreak> <case> OMGWTF <linebreak> <statement> <linebreak> OIC   
        """
        if self.expect("Variable Identifier") or self.expect("Integer Literal") or self.expect("Float Literal") or self.expect("String Delimiter") or self.expect("Boolean Literal"):
            it_value = self.current_token()[0]
            self.consume(self.current_token()[1])
            if self.expect("Linebreak"):
                self.consume("Linebreak")
                if self.expect("Switch-Case Delimiter", "WTF?"):
                    self.consume("Switch-Case Delimiter", "WTF?")
                    if self.expect("Linebreak"):
                        self.consume("Linebreak")
                        cases = self.case()
                        if self.expect("Default Keyword", "OMGWTF"):
                            self.consume("Default Keyword", "OMGWTF")
                            if self.expect("Linebreak"):
                                self.consume("Linebreak")
                                default = self.statement()
                                if self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                                    self.consume("If-Then/Switch-Case Delimiter", "OIC")
                                    return {"type": "Switch-Case", "it_value": it_value, "cases": cases, "default": default}
                                else:
                                    raise SyntaxError(f"Expected 'OIC' after default case, found {self.current_token()}")
                            else:
                                raise SyntaxError(f"Expected a linebreak after 'OMGWTF', found {self.current_token()}")
                        else:
                            raise SyntaxError(f"Expected 'OMGWTF' after cases, found {self.current_token()}")
                    else:
                        raise SyntaxError(f"Expected a linebreak after 'WTF?', found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected 'WTF?' after variable identifier, found {self.current_token()}")

    def case(self):
        """
        <case> ::= OMG <literal> <linebreak> <statement> <linebreak> | 
                   OMG <literal> <linebreak> <statement> <linebreak> <case>
        """
        cases = []
        while self.expect("Case Keyword", "OMG"):
            self.consume("Case Keyword", "OMG")
            if self.expect("Integer Literal") or self.expect("Float Literal") or self.expect("String Delimiter") or self.expect("Boolean Literal"):
                literal = self.literal()
                if self.expect("Linebreak"):
                    self.consume("Linebreak")
                    body = self.statement()
                    cases.append({"type": "Case", "literal": literal, "body": body})
                else:
                    raise SyntaxError(f"Expected a linebreak after literal, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a literal after 'OMG', found {self.current_token()}")
        return cases
    
    def literal(self):
        '''Should return the literals for the switch-cases'''
        if self.expect("Integer Literal") or self.expect("Float Literal") or self.expect("String Delimiter") or self.expect("Boolean Literal"):
            if self.expect("Integer Literal"):
                literal = self.current_token()[0]
                self.consume("Integer Literal")
            elif self.expect("Float Literal"):
                literal = self.current_token()[0]
                self.consume("Float Literal")
            elif self.expect("String Delimiter"):
                self.consume("String Delimiter")
                if self.expect("String Literal"):
                    literal = self.current_token()[0]
                    self.consume("String Literal")
                    if self.expect("String Delimiter"):
                        self.consume("String Delimiter")
            elif self.expect("Boolean Literal"):
                literal = self.current_token()[0]
                self.consume("Boolean Literal")
                
        return literal

    def statement(self):
        """
        <statement> ::= <print> <linebreak> <statement>      | <print> |
                        <scan> <linebreak> <statement>       | <scan> |
                        <assignment> <linebreak> <statement> | <assignment> |
                        <expr> <linebreak> <statement>       | <expr> |
                        <concatenate> <linebreak> <statement>| <concatenate> |
                        <typecast> <linebreak> <statement>   | <typecast> |
                        <retype> <linebreak> <statement>     | <retype> |
                        <funccall> <linebreak> <statement>   | <funccall> |
                        <loop> <linebreak> <statement>       | <loop> |
                        <ifthen> <linebreak> <statement>     | <ifthen> |
                        <switchcase> <linebreak> <statement> | <switchcase> |
                        <comment> <statement>                | <comment> |
                        <multcomment> <statement>            | <multcomment>
        """
        statements = []
        while self.current_token() and not self.expect("Code Delimiter", "KTHXBYE"):
            if self.current_token()[1] == "Function Delimiter":
                self.parse_error(f"Syntax Error: Unexpected function definition: {self.current_token()} before KTHXBYE.")
                # raise SyntaxError(f"Unexpected function definition: {self.current_token()} before KTHXBYE.")
            else:
                expr_ops = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
                if self.expect("Output Keyword"):  #VISIBLE keyword (print)
                    statements.append(self.print())
                elif self.expect("Input Keyword"):  #GIMMEH keyword (scan)
                    statements.append(self.scan())
                elif self.expect("Variable Identifier") and self.next_token()[1] == "Variable Reassignment":  #(assignment)
                    statements.append(self.assignment())
                elif self.current_token()[1] in expr_ops:  #(expr)
                    statements.append(self.expr())
                elif self.expect("Concatenation"):  #SMOOSH keyword (concatenate)
                    statements.append(self.concatenate())
                elif self.expect("Typecasting Declaration", "MAEK"):  #MAEK keyword (typecast)
                    statements.append(self.typecast())
                elif self.expect("Variable Identifier") and self.next_token()[1] == "Typecasting Reassignment":  #IS NOW A keyword (retype)
                    statements.append(self.retype())
                elif self.expect("Function Call Delimiter", "I IZ"):  #I IZ keyword (funccall)
                    statements.append(self.funccall())
                elif self.expect("Loop Delimiter", "IM IN YR"):  #IM IN YR keyword (loop)
                    statements.append(self.loop())
                elif self.expect("If-Then Delimiter", "O RLY?"):  #O RLY keyword (ifthen)
                    statements.append(self.ifthen())
                elif self.expect("Variable Identifier") and self.next_next_token()[1] == "Switch-Case Delimiter":  #WTF? keyword (switchcase)
                    statements.append(self.switchcase())
                elif self.expect("Comment"):  #BTW keyword (comment)
                    self.comment()
                elif self.expect("Multiline Comment Delimiter", "OBTW"):  #OBTW keyword (multiline comment)
                    self.multcomment()
                elif self.expect("Linebreak"):
                    self.consume("Linebreak")
                    if self.expect("Loop Delimiter", "IM OUTTA YR") or self.expect("Break/Return", "GTFO"):
                        if self.expect("Break/Return", "GTFO"):
                            self.consume("Break/Return")
                            if self.expect("Linebreak"):
                                self.consume("Linebreak")
                        break
                elif self.expect("Else If Keyword", "MEBBE"):
                    return statements
                elif self.expect("Else Keyword", "NO WAI"):
                    return statements
                elif self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                    return statements
                elif self.expect("Default Keyword", "OMGWTF"):
                    return statements
                elif self.expect("Multiline Comment Delimiter", "TLDR"):
                    self.parse_error(f"Syntax Error: Unexpected TLDR found at line {self.current_token()[2]}, no OBTW found before it.")
                else:
                    self.parse_error(f"Syntax Error: Unexpected token in statement: {self.current_token()}, with next token {self.next_token()}")
                
        if self.current_token() is None:
            self.parse_error("Syntax Error: No KTHXBYE found, missing code delimiter.")
            # raise SyntaxError("No KTHXBYE found, missing code delimiter.")
        else:
            #KTHBYE should be what's next in the tokens list since we've exhausted all statements
            return statements

    def print(self):
        """
        <print> ::= VISIBLE <operand> <printext> |
                    VISIBLE <expr> <printext> 
                    VISIBLE <operand> |
                    VISIBLE <expr> |
        BONUS: optional '!' at the end of the print statement to suppress newline   
        """
        self.consume("Output Keyword")
        
        operand = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        #NOTE: noob is a special case, it's essentially the same as None in Python or NULL in C 

        expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]

        suppress_newline = False
        value = []
        if self.current_token()[1] in operand or self.current_token()[0] == "NOOB":
            value.append(self.operand())
            value.extend(self.printext())
            if self.expect("Suppression Operator"):
                self.consume("Suppression Operator")
                suppress_newline = True
        elif self.current_token()[1] in expression:
            value.append(self.expr())
            value.extend(self.printext())
            if self.expect("Suppression Operator"):
                self.consume("Suppression Operator")
                suppress_newline = True
        elif self.expect("Concatenation"):
            value.append(self.concatenate())
            value.extend(self.printext())
            if self.expect("Suppression Operator"):
                self.consume("Suppression Operator")
                suppress_newline = True
        else:
            self.parse_error(f"Syntax Error: Expected an operand or expression, found {self.current_token()}")
            # raise SyntaxError(f"Expected an operand or expression, found {self.current_token()}")
        
        return {
            "type": "Print",
            "value": value,
            "suppress_newline": suppress_newline
        }

    def operand(self):
        """
        <operand> ::= varident | <literal>
        <literal> ::= numbr | numbar | yarn | troof | noob
        """
        #special case: String Literal
        if self.expect("String Delimiter"):
            self.consume("String Delimiter")
            lexeme = self.current_token()[0]
            classification = self.current_token()[1]
            self.consume(classification, lexeme)
            self.consume("String Delimiter")
            return {"type": "Operand", "value": lexeme, "classification": classification}
        #all other literals, varident, and noob
        else:
            lexeme = self.current_token()[0]
            classification = self.current_token()[1]
            self.consume(classification, lexeme)
            return {"type": "Operand", "value": lexeme, "classification": classification}
    
    def expr(self):
        """
        <expr> ::= <math> | <boolean> | <comparison> | <relational> | <concatenate>
        """
        math = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min"]
        boolean = ["And", "Or", "Xor", "Not", "Any", "All"]
        comparison = ["Equal", "Not Equal"]
        comparison_op = ["Integer Literal", "Float Literal", "Variable Identifier"]
        relational = ["Min", "Max"]

        if self.current_token()[1] in math:
            return self.math()
        elif self.current_token()[1] in boolean:
            return self.boolean()
        elif self.current_token()[1] in comparison:
            # NOTE: <comparison> has some similarities with <relational> so we must account for that
            operator = self.current_token()[1]
            self.consume(operator)
            if self.current_token()[1] in comparison_op and self.next_token()[1] == "Another One Keyword" and self.next_next_token()[1] in comparison_op:
                return self.comparison(operator)
            elif self.current_token()[1] in comparison_op and self.next_token()[1] == "Another One Keyword" and self.next_next_token()[1] in relational:
                return self.relational(operator)
        elif self.expect("Concatenation"):
            return self.concatenate()
        else:
            self.parse_error(f"Syntax Error: Expected an expression, found {self.current_token()}")
            # raise SyntaxError(f"Expected an expression, found {self.current_token()}")
        
    def printext(self):
        """
        <printext> ::=  + <operand> <printext> |
                        + <expr> <printext> |
                        + <operand> |
                        + <expr>
        """
        operands = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]

        concat = []
        while self.expect("Concatenation Operator", "+"):
            self.consume("Concatenation Operator", "+")  #Consume "+"
            if self.current_token()[1] in operands or self.current_token()[0] == "NOOB":
                concat.append(self.operand())
            elif self.current_token()[1] in expression:
                concat.append(self.expr())
            else:
                self.parse_error(f"Syntax Error: Expected an operand or expression, found {self.current_token()}")
                # raise SyntaxError(f"Expected an operand or expression, found {self.current_token()}")
        return concat
    
    def scan(self):
        """
        <scan> ::= GIMMEH varident
        """
        self.consume("Input Keyword")
        if self.expect("Variable Identifier"):
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            return {"type": "Scan", "variable": varident}
        else:
            self.parse_error(f"Syntax Error: Expected a variable identifier after 'GIMMEH', found {self.current_token()}")
            # raise SyntaxError(f"Expected a variable identifier after 'GIMMEH', found {self.current_token()}")
        
    def comment(self):
        """
        <comment> ::= BTW <linebreak>
        """
        self.consume("Comment")
        self.consume("Linebreak")
    
    def multcomment(self):
        """
        <multcomment> ::= <linebreak> OBTW commentstr1 <linebreak> commentstr2 <linebreak> TLDR <linebreak>
        NOTE: commentstr1 and commentstr2 are already ignored during tokenization (Lexical Analysis)
        """
        obtw_line = self.current_token()[2]
        self.consume("Multiline Comment Delimiter", "OBTW")
        self.consume("Linebreak")
        while not self.expect("Multiline Comment Delimiter"):
            if self.current_token() is None:
                self.parse_error(f"Syntax Error: Missing TLDR for OBTW at line {obtw_line} making the multiline comment unclosed.")
                # raise SyntaxError(f"Missing TLDR for OBTW at line {obtw_line} making the multiline comment unclosed.")
            self.consume("Linebreak")
        self.consume("Multiline Comment Delimiter", "TLDR")
        self.consume("Linebreak")

    def assignment(self):
        """
        <assignment> ::= varident R <operand> | varident R <expr>
                            varident R <typecast>
        """
        if self.expect("Variable Identifier"):
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            self.consume("Variable Reassignment")
            operand = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
            expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
            if self.current_token()[1] in operand or self.current_token()[0] == "NOOB":
                value = self.operand()
            elif self.current_token()[1] in expression:
                value = self.expr()
            elif self.expect("Concatenation"):
                value = self.concatenate()
            elif self.expect("Typecasting Declaration"):
                value = self.typecast()
            else:
                self.parse_error(f"Syntax Error: Expected an operand or expression, found {self.current_token()}")
                # raise SyntaxError(f"Expected an operand or expression, found {self.current_token()}")
            self.consume("Linebreak")
            return {"type": "Assignment", "variable": varident, "value": value}
        else:
            self.parse_error(f"Syntax Error: Expected a variable identifier after 'R', found {self.current_token()}")
            # raise SyntaxError(f"Expected a variable identifier, found {self.current_token()}")
    
    def math(self):
        """
        <math> ::=  <mathoperator> OF <mathext> AN <mathext>
        """
        math_ops = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min"]
        if self.current_token()[1] in math_ops:
            operator = self.current_token()[1]
            self.consume(operator)
            self.consume("First One Keyword", "OF")
            left = self.mathext()
            self.consume("Another One Keyword", "AN")
            right = self.mathext()
            return {
                "type": "Math",
                "operator": operator,
                "left": left,
                "right": right
            }
        else:
            self.parse_error(f"Syntax Error: Expected a Math Operator, but found {self.current_token()}")
            # raise SyntaxError(f"Expected a Math Operator, but found {self.current_token()}")

    def mathext(self):
        """
       <mathext> ::= <operand> | <math>
        """
        literal_varident = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        math_ops = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min"]
        if self.current_token()[1] in literal_varident:
            return self.operand()
        elif self.current_token()[1] in math_ops:
            return self.math()
        else:
            self.parse_error(f"Syntax Error: Unexpected token {self.current_token()} while parsing mathext")
    
    def boolean(self):
        """
        <boolean> ::= <booloperator> OF <boolext> AN <boolext> |
                      <boolmulti> OF <boolext> <boolmore> MKAY 
        """
        bool_ops = ["And", "Or", "Xor"]
        bool_multi = ["All", "Any"]

        if self.current_token()[1] in bool_ops:
            operator = self.current_token()[1]
            self.consume(operator)
            self.consume("First One Keyword", "OF")
            left = self.boolext()
            self.consume("Another One Keyword", "AN")
            right = self.boolext()
            return {
                "type": "Boolean",
                "operator": operator,
                "left": left,
                "right": right
            }
        elif self.current_token()[1] in bool_multi:
            operator = self.current_token()[1]
            self.consume(operator)
            self.consume("First One Keyword", "OF")
            operands = [self.boolext()]
            operands.extend(self.boolmore())
            return {
                "type": "Boolean",
                "operator": operator,
                "operands": operands
            }
        elif self.expect("Not", "NOT"):
            self.consume("Not", "NOT")
            operand = self.boolext()
            return {
                "type": "Boolean",
                "operator": "Not",
                "operand": operand
            }
        else:
            self.parse_error(f"Syntax Error: Expected a boolean operator, found {self.current_token()}")

    def boolext(self):
        """
        <boolext> ::= <operand> | <boolean> | NOT <operand> | NOT <boolean>
        """
        literal_varident = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        bool_ops = ["And", "Or", "Xor"]
        if self.current_token()[1] in literal_varident:
            return self.operand()
        elif self.current_token()[1] in bool_ops:
            return self.boolean()
        elif self.expect("Not", "NOT"):
            self.consume("Not", "NOT")
            operand = self.boolext()
            return {
                "type": "Boolean",
                "operator": "Not",
                "operand": operand
            }
        else:
            self.parse_error(f"Syntax Error: Unexpected token {self.current_token()} while parsing boolext")
            # raise SyntaxError(f"Unexpected token {self.current_token()} while parsing boolext")

    def boolmore(self):
        """
        <boolmore> ::= AN <boolext> | AN <boolext> <boolmore>
        """
        self.consume("Another One Keyword", "AN")
        operands = []
        while self.current_token and not self.expect("Function Call Delimiter", "MKAY"):
            operands.append(self.boolext())
            if self.expect("Another One Keyword", "AN"):
                self.consume("Another One Keyword", "AN")
        if self.current_token() is None:
            self.parse_error("Syntax Error: No MKAY found, missing boolean delimiter.")
        else:
            self.consume("Function Call Delimiter", "MKAY")
            return operands
        
    def comparison(self, operator):
        """
        <comparison> ::= <compoperator> numbr AN numbr |
                         <compoperator> numbar AN numbar
        <compoperator> ::= BOTH SAEM | DIFFRINT
                            (equal)     (not equal)
        """
        left = self.operand()
        self.consume("Another One Keyword", "AN")
        right = self.operand()
        return {
            "type": "Comparison",
            "operator": operator,
            "left": left,
            "right": right
        }
    
    def relational(self, operator):
        """
        <relational> ::= <compoperator> numbr <reloperator> numbr AN numbr |
                         <compoperator> numbar <reloperator> numbar AN numbar
        <compoperator> ::= BOTH SAEM | DIFFRINT
                            (equal)    (not equal)
        <reloperator> ::= BIGGR OF | SMALLR OF
                            (max)    (min)
        """
        left = self.operand()
        self.consume("Another One Keyword")
        rel_op = None
        if operator == "Equal":
            if self.expect("Max"):
                rel_op = "Greater Than or Equal"
                self.consume("Max")
            elif self.expect("Min"):
                rel_op = "Less Than or Equal"
                self.consume("Min")
            else:
                self.parse_error(f"Syntax Error: Expected a relational operator, found {self.current_token()}")
                # raise SyntaxError(f"Expected a relational operator, found {self.current_token()}")
        elif operator == "Not Equal":
            if self.expect("Max"):
                rel_op = "Less Than"
                self.consume("Max")
            elif self.expect("Min"):
                rel_op = "Greater Than"
                self.consume("Min")
            else:
                self.parse_error(f"Syntax Error: Expected a relational operator, found {self.current_token()}")
                # raise SyntaxError(f"Expected a relational operator, found {self.current_token()}")
        else:
            self.parse_error(f"Syntax Error: Expected a comparison operator, found {self.current_token()}")
            # raise SyntaxError(f"Expected a comparison operator, found {self.current_token()}")

        self.consume("First One Keyword")
        left_again = self.operand()
        if left == left_again:
            self.consume("Another One Keyword", "AN")
            right = self.operand()
            return {
                "type": "Relational",
                "operator": rel_op,
                "left": left,
                "right": right
            }
        else:
            self.parse_error(f"Syntax Error: Expected the same operand in relational operation, found {left} and {left_again}")
            # raise SyntaxError(f"Expected the same operand in relational operation, found {left} and {left_again}")

    def concatenate(self):
        """
        <concatenate> ::= SMOOSH <operand> <concatexten>
        """
        self.consume("Concatenation")
        value = [self.operand()]
        value.extend(self.concatexten())
        return {
            "type": "Concatenation",
            "value": value
        }

    def concatexten(self):
        """
        <concatexten> ::= AN <operand> | AN <operand> <concatexten>
        """
        operand = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
        
        concat = []
        while self.expect("Another One Keyword", "AN"):
            self.consume("Another One Keyword", "AN")
            if self.current_token()[1] in operand or self.current_token()[0] == "NOOB":
                concat.append(self.operand())
            elif self.current_token()[1] in expression:
                concat.append(self.expr())
        return concat

    def variable(self):
        """
        <variable> ::= WAZZUP <linebreak> <vardef> <linebreak> BUHBYE | emptystr
        """
        #Parse variable definitions between 'WAZZUP' and 'BUHBYE'
        if self.expect("Variable Delimiter", "WAZZUP"):
            self.consume("Variable Delimiter", "WAZZUP")
            self.consume("Linebreak")
            vardefs = []
            vardefs.extend(self.vardef())
            self.consume("Variable Delimiter", "BUHBYE")
            return vardefs
        else:
            return []

    def vardef(self):
        """
        <vardef>   ::= I HAS A varident <linebreak> |
                       I HAS A varident ITZ <literal> <linebreak> | 
                       I HAS A varident ITZ varident <linebreak>| 
                       I HAS A varident ITZ <expr> <linebreak>
        """
        vardefs = []
        while self.current_token() and not self.expect("Variable Delimiter", "BUHBYE"):
            #I HAS A
            if self.expect("Variable Declaration"):
                self.consume("Variable Declaration")
                #varident
                if self.expect("Variable Identifier"):
                    varident = self.current_token()[0]
                    self.consume("Variable Identifier")
                    #ITZ
                    if self.expect("Variable Assignment"):
                        self.consume("Variable Assignment")
                        
                        literal_varident = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
                        expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
                        
                        #varident | <literal>
                        if self.current_token()[1] in literal_varident or self.current_token()[0] == "NOOB":
                            value = self.operand()
                            if self.expect("Comment"):
                                self.comment()
                            else:
                                self.consume("Linebreak")
                            vardefs.append({
                                "type": "Variable Definition",
                                "name": varident,
                                "value": value
                            })
                            self.symbol_table[varident] = (value["classification"], value["value"])
                        #<expr>
                        elif self.current_token()[1] in expression:
                            value = self.expr()
                            if self.expect("Comment"):
                                self.comment()
                            else:
                                self.consume("Linebreak")
                            vardefs.append({
                                "type": "Variable Definition",
                                "name": varident,
                                "value": value
                            })
                            #if expr, don't add to symbol table yet, it will be added during evaluation (semantic_analyzer part)
                        else:
                            self.parse_error(f"Syntax Error: Expected a literal, expression or linebreak, found {self.current_token()}")
                    elif self.expect("Comment"):
                        self.comment()
                        vardefs.append({
                            "type": "Variable Declaration",
                            "name": varident,
                            "value": {"type": "Variable Definition", "value": "NOOB", "classification": "Type Literal"}
                        })
                        self.symbol_table[varident] = ("Type Literal", "NOOB")
                    elif self.expect("Linebreak"):
                        self.consume("Linebreak")
                        vardefs.append({
                            "type": "Variable Declaration",
                            "name": varident,
                            "value": {"type": "Variable Definition", "value": "NOOB", "classification": "Type Literal"}
                        })
                        self.symbol_table[varident] = ("Type Literal", "NOOB")
                    else:
                        self.parse_error(f"Syntax Error: Expected a variable assignment or linebreak, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
                else:
                    self.parse_error(f"Syntax Error: Expected a variable identifier, found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
            elif self.expect("Comment"):
                self.comment()
            elif self.expect("Multiline Comment Delimiter", "OBTW"):
                self.multcomment()
            elif self.expect("Linebreak"):
                self.consume("Linebreak")
            else:
                self.parse_error(f"Syntax Error: Expected a comment, variable declaration, or BUHBYE but found {self.current_token()[0]}-{self.current_token()[1]} at line {self.current_token()[2]}.")
        
        if self.current_token() is None:
            self.parse_error("Syntax Error: No BUHBYE found, missing variable delimiter.")
        else:
            return vardefs

        
