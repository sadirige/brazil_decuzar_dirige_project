'''
This part is the file containing the lexical analyzer producing tokens from the input lolcode on the text editor.
'''
import re

regexDictionary = {
    r'\+': "Concatenation Operator",
    r'[\"][^\"]*[\"]': "String Literal",
    r'\b-?[0-9]+\.[0-9]+\b': "Float Literal",
    r'\b-?[0-9]+\b': "Integer Literal",
    r'\b(WIN|FAIL)\b': "Boolean Literal",
    r'\b(NUMBR|NUMBAR|YARN|TROOF|NOOB)\b': "Type Literal",
    r'\bHAI\b': "Code Delimiter",
    r'\bKTHXBYE\b': "Code Delimiter",
    r'\bWAZZUP\b': "Variable Delimiter",
    r'\bBUHBYE\b': "Variable Delimiter",
    r'\bBTW\b': "Comment",
    r'\bOBTW\b': "Multiline Comment Delimiter",
    r'\bTLDR\b': "Multiline Comment Delimiter",
    r'\bI HAS A\b': "Variable Declaration",
    r'\bITZ\b': "Variable Assignment",
    r'\bR\b': "Variable Reassignment",
    r'\bSUM\b': "Add",
    r'\bDIFF\b': "Subtract",
    r'\bPRODUKT\b': "Multiply",
    r'\bQUOSHUNT\b': "Divide",
    r'\bMOD\b': "Modulo",
    r'\bBIGGR\b': "Max",
    r'\bSMALLR\b': "Min",
    r'\bBOTH\b': "And",
    r'\bEITHER\b': "Or",
    r'\bWON\b': "Xor",
    r'\bNOT\b': "Not",
    r'\bANY\b': "Any",
    r'\bALL\b': "All",
    r'\bOF\b': "First One Keyword",
    r'\bBOTH SAEM\b': "Equal",
    r'\bDIFFRINT\b': "Not Equal",
    r'\bSMOOSH\b': "Concatenation",
    r'\bMAEK\b': "Typecasting Declaration",
    r'\bA\b': "Typecasting Assignment",
    r'\bIS NOW A\b': "Typecasting Reassignment",
    r'\bVISIBLE\b': "Output Keyword",
    r'\bGIMMEH\b': "Input Keyword",
    r'\bO RLY\?\b': "If-Then Delimiter",
    r'\bYA RLY\b': "If Keyword",
    r'\bMEBBE\b': "Else If Keyword",
    r'\bNO WAI\b': "Else Keyword",
    r'\bOIC\b': "If-Then/Switch-Case Delimiter",
    r'\bWTF\?\b': "Switch-Case Delimiter",
    r'\bOMG\b': "Case Keyword",
    r'\bOMGWTF\b': "Default Keyword",
    r'\bIM IN YR\b': "Loop Delimiter",
    r'\bUPPIN\b': "Increment Keyword",
    r'\bNERFIN\b': "Decrement Keyword",
    r'\bYR\b': "Variable Call",
    r'\bTIL\b': "Loop Until",
    r'\bWILE\b': "Loop While",
    r'\bIM OUTTA YR\b': "Loop Delimiter",
    r'\bHOW IZ I\b': "Function Delimiter",
    r'\bIF U SAY SO\b': "Function Delimiter",
    r'\bGTFO\b': "Break/Return",
    r'\bFOUND YR\b': "Return With Value",
    r'\bI IZ\b': "Function Call Delimiter",
    r'\bMKAY\b': "Function Call Delimiter",
    r'\bAN\b': "Another One Keyword",
    r'\b[a-zA-Z][a-zA-Z0-9_]*\b': "Variable Identifier"
}

def tokenize(lolcode):
    tokens = []
    line_number = 1
    multiline_comment = False #flag to check if the current line is part of a multiline comment

    for line in lolcode: 
        line = line.strip()
        match_found = False 

        while line:
            #iterate through each regex to check if certain sequence of characters in line match them
            i = 0
            while i < len(regexDictionary):
                regex, classification = list(regexDictionary.items())[i]
                match = re.match(regex, line)      
                if match:
                    i = 0  #once there's a match, reset i so the next sequence of characters will be checked from start to finish of the regexDictionary
                    match_found = True
                    lexeme = match.group(0)

                    #if TLDR is found in line, just set flag of multiline comment to false and continue to check if there are other lexemes in the same line as TLDR
                    #this is not allowed based on the project specs, but it's not the job of the lexical analyzer to check this, it's the syntax analyzer's
                    if lexeme == "TLDR":
                        tokens.append((lexeme, classification, line_number))
                        multiline_comment = False
                    
                    #once OBTW is found, set multiline comment flag to true, then ignore the rest of the line (characters after OBTW)
                    #add OBTW to tokens, it won't be shown in the Lexemes part of the GUI but it's necessary for the syntax analysis
                    elif multiline_comment == False and lexeme == "OBTW":
                        tokens.append((lexeme, classification, line_number))
                        multiline_comment = True
                        line = ""
                        break

                    #if multiline comment flag is true, ignore line since it means it's between OBTW and TLDR
                    elif multiline_comment:
                        line = ""
                        break

                    #single line comment, ignore additional characters after BTW
                    elif lexeme == "BTW":
                        tokens.append((lexeme, classification, line_number))
                        line = ""
                        break
                    
                    #if a string literal was matched, break it down into delimiters (quotation symbols) and the actual string value
                    elif classification == "String Literal":
                        tokens.append((lexeme[0], "String Delimiter", line_number))
                        tokens.append((lexeme[1:-1], classification, line_number))
                        tokens.append((lexeme[-1], "String Delimiter", line_number))

                    #other regex matches can just be added to tokens without additional instructions
                    else:
                        tokens.append((lexeme, classification, line_number))

                    line = line[len(lexeme):].strip() #remove parts of the line that have already been matched
                else:
                    i += 1

            #Check if there are any unrecognized parts left in the line after iterating through all regex
            if not match_found:
                error_message = "Lexical Error: Unrecognized token: " + line.strip() + " at line " + str(line_number) + "."
                tokens.append((error_message, "Lexical Error", line_number))
            
            match_found = False #Reset flag for next line

        #Add newline/linebreak after every line, this won't be shown in the Lexemes part GUI, but it will be used for Syntax Analysis
        tokens.append(("\n", "Linebreak", line_number))
        line_number += 1
    
    return tokens
