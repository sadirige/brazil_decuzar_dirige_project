'''
This part is the file containing the syntax analyzer parsing through the lexemes and checking if the syntax is correct.
'''

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.symbol_table = {}      #Variable names and their values, including function calls
        self.ast = {
            "pre_programext": [],      #Comments, Multiline Comments, and Function Definitions before "HAI"
            "main_program": None,   #Statements between "HAI" and "KTHXBYE"
            "post_programext": []      #Comments, Multiline Comments, and Function Definitions after "KTHXBYE"
        }

    def current_token(self):
        return self.tokens[self.current_index] if self.current_index < len(self.tokens) else None

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
            self.ast["main_program"] = {"type": "Program", "body": [self.statement()]}
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
            if program_delimiter == "HAI":
                self.ast["pre_programext"].append(self.funcdef())
            else:
                self.ast["post_programext"].append(self.funcdef())

        elif self.expect("Comment"):
            self.consume("Comment")
            self.consume("Linebreak")

        elif self.expect("Multiline Comment Delimiter", "OBTW"):
            obtw_line = self.current_token()[2]
            self.consume("Multiline Comment Delimiter")
            self.consume("Linebreak")
            while not self.expect("Multiline Comment Delimiter"):
                if self.current_token() is None:
                    raise SyntaxError(f"Missing TLDR for OBTW at line {obtw_line} making the multiline comment unclosed.")
                self.consume("Linebreak")
            self.consume("Multiline Comment Delimiter")
            self.consume("Linebreak")

        elif self.expect("Linebreak"):
            self.consume("Linebreak")

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
        if self.expect("Output Keyword"):  #VISIBLE keyword (print)
            return self.print()
        elif self.expect("Linebreak"):
            self.consume("Linebreak")
            return None
        else:
            raise SyntaxError(f"Unexpected token in statement: {self.current_token()}")

    def print(self):
        """
        <print> ::= VISIBLE <operand> |
                    VISIBLE <expr> | 
                    VISIBLE <operand> <printext> |
                    VISIBLE <expr> <printext> 
        """
        self.consume("Output Keyword")
        value = None
        #if the operand is a string literal
        if self.expect("String Delimiter"):
            self.consume("String Delimiter")
            value = self.current_token()[0]  #The value to print
            self.symbol_table["IT"] = ("str", value)
            self.consume("String Literal")  # Consume the literal value
            self.consume("String Delimiter")
            self.consume("Linebreak")
        return {
            "type": "Print",
            "value": value
        }

    def variable(self):
        """
        <variable> ::= WAZZUP <linebreak> <vardef> <linebreak> BUHBYE | emptystr
        <vardef>   ::= I HAS A varident <linebreak> |
                       I HAS A varident ITZ <literal> <linebreak> | 
                       I HAS A varident ITZ varident <linebreak>| 
                       I HAS A varident <expr> <linebreak>
        """
        nodes = []
