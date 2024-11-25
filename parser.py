'''
This part is the file containing the syntax analyzer parsing through the lexemes and checking if the syntax is correct.
'''

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.symbol_table = {}      #Variable names and their values, including function calls
        self.ast = {
            "functions": [],        #Comments, Multiline Comments, and Function Definitions before "HAI" and after "KTHXBYE" (functions are what's actually stored, comments are ignoreds)
            "main_program": None,   #Statements between "HAI" and "KTHXBYE"
        }

    def current_token(self):
        return self.tokens[self.current_index] if self.current_index < len(self.tokens) else None
    
    def next_token(self):
        return self.tokens[self.current_index + 1] if self.current_index + 1 < len(self.tokens) else None

    def expect(self, classification, lexeme=None):
        token = self.current_token()
        return token and token[1] == classification and token[0] == lexeme if lexeme else token and token[1] == classification

    def consume(self, classification, lexeme=None):
        if self.expect(classification, lexeme):
            self.current_index += 1
        else:
            raise SyntaxError(f"Expected {classification} {lexeme} but found {self.current_token()} at line {self.current_token()[2]}")

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
            if self.expect("Variable Delimiter", "WAZZUP"):
                self.ast["main_program"] = {"type": "Program", "variables": self.variable(), "statements": self.statement()}
            else:
                self.ast["main_program"] = {"type": "Program", "body": self.statement()}
            self.consume("Code Delimiter", "KTHXBYE")
            self.consume("Linebreak")
        else:
            return self.symbol_table, ("Syntax Error", "No main program found (missing 'HAI').")

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
            raise SyntaxError(f"Unexpected TLDR found at line {self.current_token()[2]}, no OBTW found before it.")

        else:
            if program_delimiter == "HAI":
                raise SyntaxError(f"Unexpected token before 'HAI': {self.current_token()}")
            else:
                raise SyntaxError(f"Unexpected token after 'KTHXBYE': {self.current_token()}")
        
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
                if self.expect("Linebreak"):
                    self.consume("Linebreak")
                    statements = self.funcbody()
                    self.consume("Function Delimiter", "IF U SAY SO")
                    return {"type": "Function Definition", "name": funcident, "statements": statements}
                elif self.expect("Function Delimiter", "IF U SAY SO"):
                    self.consume("Function Delimiter", "IF U SAY SO")
                    return {"type": "Function Definition", "name": funcident, "statements": []}
                elif self.expect("Function Delimiter", "YR"):
                    self.consume("Function Delimiter", "YR")
                    param = self.param()
                    self.consume("Linebreak")
                    statements = self.funcbody()
                    self.consume("Function Delimiter", "IF U SAY SO")
                    return {"type": "Function Definition", "name": funcident, "param": param, "statements": statements}
                else:
                    raise SyntaxError(f"Expected a linebreak or 'IF U SAY SO' after function definition, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a function identifier after 'HOW IZ I', found {self.current_token()}")

    def funcbody(self):
        """
        <funcbody> ::= <statement> | <statement> <funcbody> | <funcret>
        """
        statements = []
        while not self.expect("Function Delimiter", "IF U SAY SO"):
            if self.expect("Code Delimiter", "HAI"):
                raise SyntaxError(f"Unexpected 'HAI' found before 'IF U SAY SO': {self.current_token()}")
            elif self.current_token() is None:
                raise SyntaxError("No 'IF U SAY SO' found, missing function delimiter.")
            elif self.expect("Return With Value") or self.expect("Break/Return"):
                statements.append(self.funcret())
                break
                #anything after GTFO or FOUND YR is ignored
            else:
                statements.append(self.statement())
        return statements
    
    def param(self):
        """
        <param> ::= YR varident | YR varident <paramext>
        """
        if self.expect("Function Delimiter", "YR"):
            self.consume("Function Delimiter", "YR")
            if self.expect("Variable Identifier"):
                varident = self.current_token()[0]
                self.consume("Variable Identifier")
                if self.expect("Function Delimiter", "AN"):
                    return {"type": "Parameter", "name": varident, "next": self.paramext()}
                else:
                    return {"type": "Parameter", "name": varident}
            else:
                raise SyntaxError(f"Expected a variable identifier after 'YR', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected 'YR' after 'HOW IZ I', found {self.current_token()}")

    def paramext(self):
        """
        <paramext> ::= AN YR varident | AN YR varident <paramext>
        """
        if self.expect("Another One Keyword", "AN"):
            self.consume("Another One Keyword", "AN")
            self.consume("Function Delimiter", "YR")
            if self.expect("Variable Identifier"):
                varident = self.current_token()[0]
                self.consume("Variable Identifier")
                if self.expect("Function Delimiter", "AN"):
                    return {"type": "Parameter", "name": varident, "next": self.paramext()}
                else:
                    return {"type": "Parameter", "name": varident}
            else:
                raise SyntaxError(f"Expected a variable identifier after 'YR', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected 'AN' after 'YR', found {self.current_token()}")

    def funcret(self):
        """
        <funcret> ::= FOUND YR varident | FOUND YR <expr> | GTFO
        """
        if self.expect("Return With Value"):
            self.consume("Return With Value")
            if self.expect("Variable Identifier"):
                varident = self.current_token()[0]
                self.consume("Variable Identifier")
                self.consume("Linebreak")
                return {"type": "Return", "value": varident}
            elif self.expect("Variable Identifier"):
                expr = self.expr()
                self.consume("Linebreak")
                return {"type": "Return", "value": expr}
            else:
                raise SyntaxError(f"Expected a variable identifier or expression after 'FOUND YR', found {self.current_token()}")
        elif self.expect("Break/Return"):
            self.consume("Break/Return")
            self.consume("Linebreak")
            return {"type": "Break"}
        else:
            raise SyntaxError(f"Expected 'FOUND YR' or 'GTFO', found {self.current_token()}")
        
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
                elif self.expect("Function Delimiter", "YR"):
                    param = self.param()
                    self.consume("Function Call Delimiter", "MKAY")
                    return {"type": "Function Call", "name": funcident, "param": param}
                else:
                    raise SyntaxError(f"Expected 'MKAY' or 'YR' after function identifier, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a function identifier after 'I IZ', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected 'I IZ' after 'HOW IZ I', found {self.current_token()}")

    def typecast(self):
        """
        <typecast> ::= MAEK varident <type>
        <type> ::= A TROOF | A NUMBR | A NUMBAR | YARN
        """
        self.consume("Typecast", "MAEK")
        if self.expect("Variable Identifier"):
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            if self.expect("Type"):
                type = self.current_token()[0]
                self.consume("Type")
                return {"type": "Typecast", "variable": varident, "type": type}
            else:
                raise SyntaxError(f"Expected a type after 'MAEK', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected a variable identifier after 'MAEK', found {self.current_token}")

    def retype(self):
        """
        <retype> ::= varident IS NOW <type> | varident R MAEK varident A <type>
        """
        if self.expect("Variable Identifier"):
            varident = self.current_token()[0]
            self.consume("Variable Identifier")
            if self.expect("Variable Assignment"):
                self.consume("Variable Assignment")
                if self.expect("Type"):
                    type = self.current_token()[0]
                    self.consume("Type")
                    return {"type": "Retype", "variable": varident, "type": type}
                else:
                    raise SyntaxError(f"Expected a type after 'IS NOW', found {self.current_token()}")
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
                            return {"type": "Retype", "variable": varident, "type": type}
                        else:
                            raise SyntaxError(f"Expected a type after 'A', found {self.current_token()}")
                    else:
                        raise SyntaxError(f"Expected 'A' after variable identifier, found {self.current_token()}")
                else:
                    raise SyntaxError(f"Expected a variable identifier after 'MAEK', found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected 'IS NOW' or 'MAEK' after variable identifier, found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected a variable identifier after 'MAEK', found {self.current_token()}")
        
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
                    loopop = self.loopop()
                    self.consume("Variable Call", "YR")
                    if self.expect("Variable Identifier"):
                        varident = self.current_token()[0]
                        self.consume("Variable Identifier")
                        if self.expect("Linebreak"):
                            self.consume("Linebreak")
                            body = self.loopbody()
                            self.consume("Loop Delimiter", "IM OUTTA YR")
                            self.consume("Variable Identifier", loopident)
                            return {"type": "Loop", "name": loopident, "operation": loopop, "variable": varident, "body": body}
                        else:
                            raise SyntaxError(f"Expected a linebreak after variable identifier, found {self.current_token()}")
                    else:
                        raise SyntaxError(f"Expected a variable identifier after loop operation, found {self.current_token()}")
                elif self.expect("Loop Until") or self.expect("Loop While"):
                    loopcond = self.loopcond()
                    expr = self.expr()
                    if self.expect("Linebreak"):
                        self.consume("Linebreak")
                        body = self.loopbody()
                        self.consume("Loop Delimiter", "IM OUTTA YR")
                        self.consume("Variable Identifier", loopident)
                        return {"type": "Loop", "name": loopident, "condition": loopcond, "expression": expr, "body": body}
                    else:
                        raise SyntaxError(f"Expected a linebreak after expression, found {self.current_token()}")
                else:
                    raise SyntaxError(f"Expected a loop operation or condition after variable identifier, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a variable identifier after 'IM IN YR', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected 'IM IN YR' after 'HOW IZ I', found {self.current_token()}")

    def loopbody(self):
        """
        <loopbody> ::= <statement> | <statement> <loopbody> | GTFO
        """
        statements = []
        while not self.expect("Loop Delimiter", "IM OUTTA YR"):
            if self.expect("Code Delimiter", "HAI"):
                raise SyntaxError(f"Unexpected 'HAI' found before 'IM OUTTA YR': {self.current_token()}")
            elif self.current_token() is None:
                raise SyntaxError("No 'IM OUTTA YR' found, missing loop delimiter.")
            elif self.expect("Return With Value") or self.expect("Break/Return"):
                statements.append(self.funcret())
                break
                #anything after GTFO or FOUND YR is ignored
            else:
                statements.append(self.statement())
        return statements

    def ifthen(self):
        """
        <ifthen> ::= <expr> <linebreak> O RLY? <linebreak> YA RLY <linebreak> <statement> <linebreak> OIC | 
                     <expr> <linebreak> O RLY? <linebreak> YA RLY <linebreak> <statement> <linebreak> NO WAI <linebreak> <statement> <linebreak> OIC
        """
        if self.expect("If-Then Delimiter", "O RLY"):
            self.consume("If-Then Delimiter", "O RLY")
            if self.expect("Linebreak"):
                self.consume("Linebreak")
                if self.expect("If Keyword", "YA RLY"):
                    self.consume("If Keyword", "YA RLY")
                    if self.expect("Linebreak"):
                        self.consume("Linebreak")
                        true_body = self.statement()
                        if self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                            self.consume("If-Then/Switch-Case Delimiter", "OIC")
                            return {"type": "If-Then", "condition": self.expr(), "true_body": true_body}
                        elif self.expect("Else If Keyword", "MEBBE"):
                            self.consume("Else If Keyword", "MEBBE")
                            if self.expect("Linebreak"):
                                self.consume("Linebreak")
                                false_body = self.statement()
                                if self.expect("If-Then/Switch-Case Delimiter", "OIC"):
                                    self.consume("If-Then/Switch-Case Delimiter", "OIC")
                                    return {"type": "If-Then", "condition": self.expr(), "true_body": true_body, "false_body": false_body}
                                else:
                                    raise SyntaxError(f"Expected 'OIC' after false body, found {self.current_token()}")
                            else:
                                raise SyntaxError(f"Expected a linebreak after 'MEBBE', found {self.current_token()}")
                        else:
                            raise SyntaxError(f"Expected 'OIC' or 'MEBBE' after true body, found {self.current_token()}")
                    else:
                        raise SyntaxError(f"Expected a linebreak after 'YA RLY', found {self.current_token()}")
                else:
                    raise SyntaxError(f"Expected 'YA RLY' after linebreak, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a linebreak after 'O RLY', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected 'O RLY' after 'HOW IZ I', found {self.current_token()}")

    def switchcase(self):
        """
        <switchcase> ::= WTF? <linebreak> <case> OMGWTF <linebreak> <statement> <linebreak> OIC
        """
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
                            return {"type": "Switch-Case", "cases": cases, "default": default}
                        else:
                            raise SyntaxError(f"Expected 'OIC' after default case, found {self.current_token()}")
                    else:
                        raise SyntaxError(f"Expected a linebreak after 'OMGWTF', found {self.current_token()}")
                else:
                    raise SyntaxError(f"Expected 'OMGWTF' after cases, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a linebreak after 'WTF?', found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected 'WTF?' after 'HOW IZ I', found {self.current_token()}")

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
        while self.current_token and not self.expect("Code Delimiter", "KTHXBYE"):
            if self.current_token()[1] == "Function Delimiter":
                raise SyntaxError(f"Unexpected function definition: {self.current_token()} before KTHXBYE.")
            elif self.current_token() is None:
                raise SyntaxError("No KTHXBYE found, missing code delimiter.")
            else:
                expr_ops = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]
                if self.expect("Output Keyword"):  #VISIBLE keyword (print)
                    statements.append(self.print())
                elif self.expect("Input Keyword"):  #GIMMEH keyword (scan)
                    statements.append(self.scan())
                elif self.expect("Variable Identifier"):  #(assignment)
                    statements.append(self.assignment())
                elif self.current_token()[1] in expr_ops:  #(expr)
                    statements.append(self.expr())
                elif self.expect("Concatenation"):  #SMOOSH keyword (concatenate)
                    statements.append(self.concatenate())
                elif self.expect("Typecast", "MAEK"):  #MAEK keyword (typecast)
                    statements.append(self.typecast())
                elif self.expect("Variable Identifier"):  #IS NOW A keyword (retype)
                    statements.append(self.retype())
                elif self.expect("Function Call Delimiter", "I IZ"):  #I IZ keyword (funccall)
                    statements.append(self.funccall())
                elif self.expect("Loop Delimiter", "IM IN YR"):  #IM IN YR keyword (loop)
                    statements.append(self.loop())
                elif self.expect("If-Then Delimiter", "O RLY"):  #O RLY keyword (ifthen)
                    statements.append(self.ifthen())
                elif self.expect("Switch-Case Delimiter", "WTF?"):  #WTF? keyword (switchcase)
                    statements.append(self.switchcase())
                elif self.expect("Comment"):  #BTW keyword (comment)
                    self.comment()
                elif self.expect("Multiline Comment Delimiter", "OBTW"):  #OBTW keyword (multiline comment)
                    self.multcomment()
                elif self.expect("Linebreak"):
                    self.consume("Linebreak")
                elif self.expect("Multiline Comment Delimiter", "TLDR"):
                    raise SyntaxError(f"Unexpected TLDR found at line {self.current_token()[2]}, no OBTW found before it.")
                else:
                    raise SyntaxError(f"Unexpected token in statement: {self.current_token()}")
                
        if self.current_token() is None:
            raise SyntaxError("No KTHXBYE found, missing code delimiter.")
        else:
            #KTHBYE should be what's next in the tokens list since we've exhausted all statements
            return statements

    def print(self):
        """
        <print> ::= VISIBLE <operand> <printext> |
                    VISIBLE <expr> <printext> 
                    VISIBLE <operand> |
                    VISIBLE <expr> |   
        """
        self.consume("Output Keyword")
        
        operand = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        #NOTE: noob is a special case, it's essentially the same as None in Python or NULL in C 

        expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]

        if self.current_token()[1] in operand or self.current_token()[0] == "NOOB":
            return {
                "type": "Print",
                "value": self.operand(),
                "concatenations": self.printext()
            }
        elif self.current_token()[1] in expression:
            return {
                "type": "Print",
                "value": self.expr(),
                "concatenations": self.printext()
            }
        elif self.expect("Concatenation"):
            return {
                "type": "Print",
                "value": self.concatenate(),
                "concatenations": self.printext()
            }
        else:
            raise SyntaxError(f"Expected an operand or expression, found {self.current_token()}")

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
        # relational = ["Greater Than", "Less Than", "Greater Than or Equal", "Less Than or Equal"]

        if self.current_token()[1] in math:
            return self.math()
        elif self.current_token()[1] in boolean:
            return self.boolean()
        elif self.current_token()[1] in comparison:
            # NOTE: <comparison> has some similarities with <ralational> so we must account for that
            operator = self.current_token()[1]
            self.consume(operator)
            if self.current_token()[1] in ["Integer Literal", "Float Literal", "Variable Identifier"] and self.next_token()[1] == "Another One Keyword":
                self.comparison(operator)
            elif self.current_token()[1] in ["Integer Literal", "Float Literal", "Variable Identifier"] and self.next_token()[1] in ["Min", "Max"]:
                self.relational(operator)
        elif self.expect("Concatenation"):
            return self.concatenate()
        else:
            raise SyntaxError(f"Expected an expression, found {self.current_token()}")
        
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
                raise SyntaxError(f"Expected an operand or expression, found {self.current_token()}")
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
            raise SyntaxError(f"Expected a variable identifier after 'GIMMEH', found {self.current_token()}")
        
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
                raise SyntaxError(f"Missing TLDR for OBTW at line {obtw_line} making the multiline comment unclosed.")
            self.consume("Linebreak")
        self.consume("Multiline Comment Delimiter", "TLDR")
        self.consume("Linebreak")

    def assignment(self):
        """
        <assignment> ::= varident R <operand> | varident R <expr>
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
            else:
                raise SyntaxError(f"Expected an operand or expression, found {self.current_token()}")
            self.consume("Linebreak")
            return {"type": "Assignment", "variable": varident, "value": value}
        else:
            raise SyntaxError(f"Expected a variable identifier, found {self.current_token()}")
    
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
            raise SyntaxError(f"Expected a Math Operator, but found {self.current_token()}")

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
            raise SyntaxError(f"Unexpected token {self.current_token()} while parsing mathext")
        
    def boolean(self):
        """
        <boolean> ::=   <booloperator> OF <operand> AN <operand> |
                        <boolmulti> OF <booleanexpr> <booleanext> MKAY
        <booloperator> ::= BOTH | EITHER | WON
        <boolmulti> ::= ALL | ANY
        """
        bool_ops = ["And", "Or", "Xor"]
        bool_multi = ["All", "Any"]
        if self.current_token()[1] in bool_ops:
            operator = self.current_token()[1]
            self.consume(operator)
            self.consume("First One Keyword", "OF")
            left = self.operand()
            self.consume("Another One Keyword", "AN")
            right = self.operand()
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
            left = self.booleanexpr()
            return {
                "type": "Boolean",
                "operator": operator,
                "left": left,
                "right": self.booleanext()
            }
        else:
            raise SyntaxError(f"Expected a boolean operator, found {self.current_token()}")
        
    def booleanexpr(self):
        """
        <booleanexpr> ::= <booloperator> OF <operand> AN <operand> | NOT <operand>
        """
        bool_ops = ["And", "Or", "Xor"]
        if self.current_token()[1] in bool_ops:
            operator = self.current_token()[1]
            self.consume(operator)
            self.consume("First One Keyword", "OF")
            left = self.operand()
            self.consume("Another One Keyword", "AN")
            right = self.operand()
            return {
                "type": "Boolean",
                "operator": operator,
                "left": left,
                "right": right
            }
        elif self.expect("Boolean Operator", "Not"):
            self.consume("Boolean Operator", "Not")
            operand = self.operand()
            return {
                "type": "Boolean",
                "operator": "Not",
                "operand": operand
            }
        else:
            raise SyntaxError(f"Expected a boolean operator, found {self.current_token()}")

    def booleanext(self):
        """
        <booleanext> ::= AN <booleanexpr> |
                         AN <booleanexpr> <booleanext>
        """
        self.consume("Another One Keyword", "AN")
        left = self.booleanexpr()
        if self.expect("Function Call Delimiter", "MKAY"):
            self.consume("Function Call Delimiter", "MKAY")
            return left
        else:
            return {
                "type": "Boolean",
                "operator": "All",
                "left": left,
                "right": self.booleanext()
            }
        
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
        rel_op = None
        if operator == "Equal":
            if self.expect("Max"):
                rel_op = "Greater Than or Equal"
                self.consume("Max")
            elif self.expect("Min"):
                rel_op = "Less Than or Equal"
                self.consume("Min")
            else:
                raise SyntaxError(f"Expected a relational operator, found {self.current_token()}")
        elif operator == "Not Equal":
            if self.expect("Max"):
                rel_op = "Greater Than"
                self.consume("Max")
            elif self.expect("Min"):
                rel_op = "Less Than"
                self.consume
            else:
                raise SyntaxError(f"Expected a relational operator, found {self.current_token()}")
        else:
            raise SyntaxError(f"Expected a comparison operator, found {self.current_token()}")

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
            raise SyntaxError(f"Expected the same operand in relational operation, found {left} and {left_again}")

    def concatenate(self):
        """
        <concatenate> ::= SMOOSH <operand> <concatexten>
        """
        self.consume("Concatenation")
        operand = self.operand()
        return {
            "type": "Concatenation",
            "value": operand,
            "concatenations": self.concatexten()
        }

    def concatexten(self):
        """
        <concatexten> ::= AN <operand> | AN <operand> <concatexten>
        """
        operand = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        concat = []
        while self.expect("Another One Keyword", "AN"):
            self.consume("Another One Keyword", "AN")
            if self.current_token()[1] in operand or self.current_token()[0] == "NOOB":
                concat.append(self.operand())

    def variable(self):
        """
        <variable> ::= WAZZUP <linebreak> <vardef> <linebreak> BUHBYE | emptystr
        """
        #Parse variable definitions between 'WAZZUP' and 'BUHBYE'
        if self.expect("Variable Delimiter", "WAZZUP"):
            self.consume("Variable Delimiter", "WAZZUP")
            self.consume("Linebreak")

            vardefs = []
            while not self.expect("Variable Delimiter", "BUHBYE"):
                if self.expect("Comment"):
                    self.comment()
                elif self.expect("Multiline Comment Delimiter", "OBTW"):
                    self.multcomment()
                else:
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
        while not self.expect("Variable Delimiter", "BUHBYE"):
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
                            self.consume("Linebreak")
                            vardefs.append({
                                "type": "Variable Definition",
                                "name": varident,
                                "value": value
                            })
                            # print("literal")
                            # print(value["value"])
                            self.symbol_table[varident] = (value["classification"], value["value"])
                        #<expr>
                        elif self.current_token()[1] in expression:
                            value = self.expr()
                            self.consume("Linebreak")
                            vardefs.append({
                                "type": "Variable Definition",
                                "name": varident,
                                "value": value
                            })
                            # self.symbol_table[varident] = value["value"]
                        else:
                            raise SyntaxError(f"Expected a literal, expression or linebreak, found {self.current_token()}")
                    elif self.expect("Linebreak"):
                        self.consume("Linebreak")
                        vardefs.append({
                            "type": "Variable Declaration",
                            "name": varident,
                            "value": {"type": "Variable Definition", "value": "NOOB", "classification": "Type Literal"}
                        })
                        self.symbol_table[varident] = ("Noob Keyword", "NOOB")
                    else:
                        raise SyntaxError(f"Expected a variable assignment or linebreak, found {self.current_token()}")
                else:
                    raise SyntaxError(f"Expected a variable identifier, found {self.current_token()}")
            else:
                raise SyntaxError(f"Expected a variable declaration, found {self.current_token()}")
            
        return vardefs

        
