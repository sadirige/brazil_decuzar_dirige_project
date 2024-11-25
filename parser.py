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

    def funcbody(self):
        """
        <funcbody> ::= <statement> | <statement> <funcbody> | <funcret>
        """

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
                if self.expect("Output Keyword"):  #VISIBLE keyword (print)
                    statements.append(self.print())
                elif self.expect("Input Keyword"):  #GIMMEH keyword (scan)
                    statements.append(self.scan())
                elif self.expect("Variable Identifier"):  #(assignment)
                    statements.append(self.assignment())
                elif self.expect("Concatenation"):  #SMOOSH keyword (concatenate)
                    statements.append(self.concatenate())
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
        operand = ["Variable Identifier", "Integer Literal", "Float Literal", "String Delimiter", "Boolean Literal"]
        expression = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min", "And", "Or", "Xor", "Not", "Any", "All", "Equal", "Not Equal"]

        concat = []
        while self.expect("Concatenation Operator", "+"):
            self.consume("Concatenation Operator", "+")  #Consume "+"
            if self.current_token()[1] in operand or self.current_token()[0] == "NOOB":
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
        <math> ::=  <mathoperator> OF <operand> AN <operand> |
                    <mathoperator> OF <math> AN <operand> |
        """
        math = ["Add", "Subtract", "Multiply", "Divide", "Modulo", "Max", "Min"]
        if self.current_token()[1] in math:
            operator = self.current_token()[1]
            self.consume(operator)
            self.consume("First One Keyword", "OF")
            left = self.math()
            self.consume("Another One Keyword", "AN")
            right = self.operand()
            return {
                "type": "Math",
                "operator": operator,
                "left": left,
                "right": right
            }
        else:
            return self.operand()
        
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
            vardefs = self.vardef()
            self.consume("Variable Delimiter", "BUHBYE")
            self.consume("Linebreak")
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

        
