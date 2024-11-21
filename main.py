'''
References:
1. Sample Graphical User Interface (GUI) from Project Specifications
2. OOP use of tkinter - CMSC 170 Exer 1 (8-Puzzle Game) Template
3. Allow Text editor, List of Tokens & Symbol Table to be horizontally resizable
   Allow (Text editor, List of Tokens & Symbol Table) and console_part to be vertically resizable
        https://www.geeksforgeeks.org/python-panedwindow-widget-in-tkinter/
        https://ultrapythonic.com/tkinter-panedwindow/
4. Adding hover effect on buttons
        https://www.geeksforgeeks.org/tkinter-button-that-changes-its-properties-on-hover/
5. Initializing root window to maximized state
        https://blog.finxter.com/5-best-ways-to-initialize-a-window-as-maximized-in-tkinter-python/
   Use of try except for different ways to maximize root window since some methods work on other devices while others can't
        https://stackoverflow.com/questions/15981000/tkinter-python-maximize-window
6. Add line numbers beside Text Editor for better UX
        https://stackoverflow.com/questions/16369470/tkinter-adding-line-number-to-text-widget
'''
import tkinter as tk
from tkinter import filedialog, ttk, PanedWindow
import re

symbolTable ={}
lexemeDictionary ={}

regexDictionary = {
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
    r'\bmultiline_comment\b': "Multi-line Comment Delimiter",
    r'\bTLDR\b': "Multi-line Comment Delimiter",
    r'\bI HAS A\b': "Variable Declaration",
    r'\bITZ\b': "Variable Assignment",
    r'\bR\b': "Variable Reassignment",
    r'\bSUM OF\b': "Add",
    r'\bDIFF OF\b': "Subtract",
    r'\bPRODUKT OF\b': "Multiply",
    r'\bQUOSHUNT OF\b': "Divide",
    r'\bMOD OF\b': "Modulo",
    r'\bBIGGR OF\b': "Max",
    r'\bSMALLR OF\b': "Min",
    r'\bBOTH OF\b': "And",
    r'\bEITHER OF\b': "Or",
    r'\bWON OF\b': "Xor",
    r'\bNOT\b': "Not",
    r'\bANY OF\b': "Any",
    r'\bALL OF\b': "All",
    r'\bBOTH SAEM\b': "Equal",
    r'\bDIFFRINT\b': "Not Equal",
    r'\bSMOOSH\b': "Concatenate",
    r'\bMAEK\b': "Typecast Declaration",
    r'\bA\b': "Typecast Assignment",
    r'\bIS NOW A\b': "Typecast Reassignment",
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

class InterpreterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Team (ﾉ◕ヮ◕)ﾉ*:･ﾟ LOLTERPRETER")
        self.current_file = None

        self.create_widgets()

        # Initialize window in maximized state
        self.root.update_idletasks()
        try:
            self.root.wm_attributes('-zoomed', True) # Works for Ubuntu and others, not Arch
        except:
            self.root.state('zoomed') # Works for Windows and others, not Ubuntu

        self.root.after(100, self.initialize_parts_sashpos)  # Update sash position after main loop starts

    # -----------------------------------------------------------------------------------------
    # Rendering the elements in the GUI
    # -----------------------------------------------------------------------------------------
    def create_widgets(self):
        #Create frame for top most part containing File explorer and 'LOL CODE Interpreter' text
        top_part = tk.Frame(root)
        top_part.grid(row=0, column=0, columnspan=2, sticky="ew")
        top_part.grid_columnconfigure(0, weight=1) #Divide area between file explorer and 'LOL CODE Interpreter'
        top_part.grid_columnconfigure(1, weight=2)

        #(1) File explorer - Allows you to select a file to run. Once a file is selected, the contents of the file should be loaded into the text editor (2).
        self.file_name = tk.Label(top_part, text="(None)") #Initialize file name as (None)
        self.file_name.grid(row=0, column=0, sticky="w")
        open_file_button = tk.Button(top_part, text="Open File", command=self.select_file) #Allow user to select a file
        open_file_button.grid(row=0, column=0, sticky="e")
        self.change_on_hover(open_file_button) #Add hover effect on button

        #Place project description 'LOL CODE Interpreter' on the right side File explorer (occupying 2/3 of top part)
        proj_desc_frame = tk.Frame(top_part, bg="black")
        proj_desc_frame.grid(row=0, column=1, sticky="ew")
        proj_desc = tk.Label(proj_desc_frame, text="LOL CODE Interpreter", fg="white", bg="black") #similar to sample GUI that has dark bg
        proj_desc.pack(side="left")

        #Create vertical paned window to hold the upper and lower parts of the GUI
        #Paned windows allow user to scroll up and down or left and right to adjust the sizes of the GUI parts for better viewing
        self.vertical_pw = PanedWindow(root, orient=tk.VERTICAL)
        self.vertical_pw.grid(row=1, column=0, columnspan=3, sticky="nsew")

        #Create horizontal paned windowto hold the Text editor, list of tokens, and symbol table
        self.horizontal_pw = PanedWindow(self.vertical_pw, orient=tk.HORIZONTAL)
        
        #(2) Text editor - Allows you to view the code you want to run. The text editor should be editable, and edits done should be reflected once the code is run.
        text_editor_part = tk.Frame(self.horizontal_pw)
        self.line_numbers = tk.Text(text_editor_part, width=4, padx=5, border=0, background="lightgray", state="disabled", wrap="none")
        self.line_numbers.pack(side="left", fill="y")
        self.text_editor = tk.Text(text_editor_part, wrap="none", undo=True)
        self.text_editor.pack(side="right", fill="both", expand=True)

        #Bind events to update line numbers counting the lines in the text editor
        self.text_editor.bind("<KeyRelease>", self.update_line_numbers)
        self.text_editor.bind("<MouseWheel>", self.update_line_numbers)
        self.text_editor.bind("<ButtonRelease-1>", self.update_line_numbers)
        
        self.update_line_numbers() # Initialize line numbers
        self.horizontal_pw.add(text_editor_part) #Add text editor to the horizontal paned window so it can be resized left and right later on

        #(3) List of Tokens - This should be updated every time the Execute/Run button (5) is pressed. This should contain all the lexemes detected from the code being ran, and their classification.
        tokens_list_part = tk.Frame(self.horizontal_pw)
        tokens_label = tk.Label(tokens_list_part, text="Lexemes")
        tokens_label.pack(anchor="center")
        self.lexemes = ttk.Treeview(tokens_list_part, columns=("Lexeme", "Classification"), show="headings") #Divide area into 2 columns
        self.lexemes.heading("Lexeme", text="Lexeme", anchor="w")
        self.lexemes.heading("Classification", text="Classification", anchor="w")
        self.lexemes.pack(fill=tk.BOTH, expand=True)
        self.horizontal_pw.add(tokens_list_part) #Add to horizontal paned window

        #(4) Symbol Table - This should be updated every time the Execute/Run button (5) is pressed. This should contain all the variables available in the program being ran, and their updated values.
        symbol_table_part = tk.Frame(self.horizontal_pw)
        symbol_label = tk.Label(symbol_table_part, text="SYMBOL TABLE")
        symbol_label.pack(anchor="center")
        self.symbols = ttk.Treeview(symbol_table_part, columns=("Identifier", "Value"), show="headings") #Also divide area into 2 columns
        self.symbols.heading("Identifier", text="Identifier", anchor="w")
        self.symbols.heading("Value", text="Value", anchor="w")
        self.symbols.pack(fill=tk.BOTH, expand=True)
        self.horizontal_pw.add(symbol_table_part) #Add to horizontal paned window

        #Add the horizontal paned window to the vertical paned window (essentially this is the upper part)
        self.vertical_pw.add(self.horizontal_pw)

        #Now we create the lower part of the vertical paned window (contains execute button and console_part)
        execute_console_part_frame = tk.Frame(self.vertical_pw)
        
        #(5) Execute/Run button - This will run the code from the text editor (2).
        execute_button = tk.Button(execute_console_part_frame, text="EXECUTE", command=self.execute_code)
        execute_button.pack(fill=tk.X, pady=(0, 5))
        self.change_on_hover(execute_button) #Add hover effect on button

        #(6) console_part - Input/Output of the program should be reflected in the console_part. For variable input, you can add a separate field for user input, or have a dialog box pop up.
        console_part = tk.Text(execute_console_part_frame, height=5, wrap="word")
        console_part.pack(fill=tk.BOTH, expand=True)

        #Add the lower part to the vertical paned window
        self.vertical_pw.add(execute_console_part_frame)

        #Configure grid weights to make the different parts responsive
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=0)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

    # -----------------------------------------------------------------------------------------
    # Adds line numbers beside each line of lolcode in the text editor part
    # -----------------------------------------------------------------------------------------
    def update_line_numbers(self, event=None):
        #count total number of lines from the text editor
        total_lines = int(self.text_editor.index("end-1c").split(".")[0])
        line_numbers = "\n".join(str(i) for i in range(1, total_lines + 1))

        #update the text widget for the line numbers (that is beside the text editor)
        self.line_numbers.config(state="normal")
        self.line_numbers.delete(1.0, "end")
        self.line_numbers.insert(1.0, line_numbers)
        self.line_numbers.config(state="disabled")

    # -----------------------------------------------------------------------------------------
    # Adds hover effect on buttons
    # -----------------------------------------------------------------------------------------
    def change_on_hover(self, button):
        #change to darker color when arrow is on button
        button.bind("<Enter>", lambda _: button.config(bg="darkgray"))
        
        #go back to usual color of tkinter button when arrow is not on button
        button.bind("<Leave>", lambda _: button.config(bg="SystemButtonFace"))

    # -----------------------------------------------------------------------------------------
    # Initializes the parts of the GUI at the start of program (50-50 vertical, 33-33-33 horizontal) paned windows
    # -----------------------------------------------------------------------------------------
    def initialize_parts_sashpos(self):
        #Divide upper and lower half of GUI into 2 equal parts, set sash position to 1/2 of root window height
        height = self.root.winfo_height()
        self.vertical_pw.sash_place(0, 0, height // 2)

        #Divide the text editor, list of tokens, and symbol table into 3 equal parts, set sash position of each to 1/3 of root window width
        width = self.root.winfo_width()
        self.horizontal_pw.sash_place(0, width // 3, 0)
        self.horizontal_pw.sash_place(1, 2 * width // 3, 0)

    # -----------------------------------------------------------------------------------------
    # Allow user to select a LOL CODE file via file dialog
    # -----------------------------------------------------------------------------------------
    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("LOL CODE files", "*.lol")])
        if file_path and file_path.endswith(".lol"):
            self.current_file = file_path
            file_name = file_path.split("/")[-1]
            self.file_name.config(text=file_name) #Update (None) to the selected file name
        else:
            tk.messagebox.showerror("Invalid File", "Please select a valid .lol file.")
            return

        #Load the contents of the selected file into the text editor (this part will execute if the function did not return since file was a valid lol file)
        with open(file_path, "r") as file:
            lolcode = file.read()
            self.text_editor.delete(1.0, tk.END)
            self.text_editor.insert(tk.END, lolcode)

        #Update line numbers in text editor
        self.update_line_numbers()

    # -----------------------------------------------------------------------------------------
    # Execute the code from the text editor after pressing the execute button
    #   (3) List of Tokens – This should be updated every time the Execute/Run button (5) is pressed.
    #   (4) Symbol Table – This should be updated every time the Execute/Run button (5) is pressed. 
    # -----------------------------------------------------------------------------------------
    def execute_code(self):
        #Clear previous lexemes and symbols
        for item in self.lexemes.get_children():
            self.lexemes.delete(item)
        for item in self.symbols.get_children():
            self.symbols.delete(item)

        #Get a copy of the lolcode from the text editor
        lolcode = self.text_editor.get("1.0", tk.END).splitlines()

        #Save the current code in the text editor to the actual .lol file
        if(self.current_file is not None):
            with open(self.current_file, "w") as file:
                file.write("\n".join(lolcode))

        #1. Tokenize each line in lolcode and display in the list of tokens (lexemes) - LEXICAL ANALYSIS
        lexemes = self.lexer(lolcode)
        print(lexemes)
        self.display_lexemes(lexemes)

        #2. Convert tokens to symbol table - SYNTAX ANALYSIS
        self.checkSyntax(lexemes)
        symbolTable = self.get_var_assignments(lexemes)
        self.display_symbol(symbolTable)

    def lexer(self, lolcode):
        tokens = []
        line_number = 1
        multiline_comment = False

        for line in lolcode: 
            line = line.strip()
            match_found = False

            while line:
                i = 0
                #iterate through each regex to check if certain sequence of characters in line match them
                while i < len(regexDictionary.items()):
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

                    i += 1 #move to the next regex and check if it matches what's on the line

                #Check if there are any unrecognized parts left in the line after iterating through all regex
                if not match_found:
                    error_message = "Unrecognized token \"" + line.strip() + "\" at line " + str(line_number) + "."
                    tk.messagebox.showerror("Lexical Error", error_message)
                    # print(line.strip())
                    tokens.clear()
                    return tokens
                
                match_found = False

            #Add newline after every line, this won't be shown in the Lexemes part, but it will be used for Syntax Analysis
            tokens.append(("\n", "Newline", line_number))
            line_number += 1
        
        return tokens

    def display_lexemes(self, lexemes):
        for lexeme, classification, _ in lexemes:
            if lexeme == "\n" and classification == "Newline":
                continue
            elif lexeme == "BTW" or lexeme == "OBTW" or lexeme == "TLDR":
                continue
            else:
                #ignore newline and comments, only show the other more important lexemes
                self.lexemes.insert('', tk.END, values=(lexeme, classification))
                
    def checkSyntax(self, lexemes):
        dupeLexeme = list(lexemes)
        index = 0

        literals = ["String Literal", "Float Literal", "Integer Literal", "Boolean Literal"]
        operands = ["varident", "literal"]
        mathoperator = ["Add", "Subtract", "Multiply", "Divide", "Mod", "Max", "Min"]

        while (True):
            for i in dupeLexeme:
                flag = False 
                index = dupeLexeme.index(i)

                # -------------- Simplify tokens
                # Literals
                if (i[1] in ["String Literal", "Float Literal", "Integer Literal", "Boolean Literal"]):
                    del dupeLexeme[index]
                    dupeLexeme.insert(index, "literal")
                    flag = True

                elif(i[1] == "String Delimiter"):
                    if(dupeLexeme[index+1]=="literal" and dupeLexeme[index+2][1]=="String Delimiter"):
                        del dupeLexeme[index:(index+3)]
                        dupeLexeme.insert(index, "literal")
                        flag = True
                
                # Variable Identifier
                elif (i[1] == "Variable Identifier"):
                    del dupeLexeme[index]
                    dupeLexeme.insert(index, "varident")
                    flag = True

                # Variable Declaration
                elif (i[1] == "Variable Declaration"):
                    if(dupeLexeme[index+1]=="varident" and dupeLexeme[index+2][1]=="Variable Assignment" and dupeLexeme[index+3] in operands):
                        del dupeLexeme[index:(index+4)]
                        dupeLexeme.insert(index, "vardef")
                        flag = True
                    elif(dupeLexeme[index+1]=="varident" and dupeLexeme[index+2]=="expr"):
                        del dupeLexeme[index:(index+3)]
                        dupeLexeme.insert(index, "vardef")
                        flag = True
                    elif(dupeLexeme[index+1]=="varident"):
                        del dupeLexeme[index:(index+2)]
                        dupeLexeme.insert(index, "vardef")
                        flag = True                  

                # Print
                if (i[1] == "VISIBLE Keyword"):
                    if(dupeLexeme[index+1] in operands or 
                       dupeLexeme[index+1] == "expr"):
                        del dupeLexeme[index:index+1]
                        dupeLexeme.insert(index, "print")

                # Scan
                elif (i[1] == "Input Keyword"):
                    if(dupeLexeme[index+1]=="varident"):
                        del dupeLexeme[index:(index+1)]
                        dupeLexeme.insert(index, "scan")
                        flag = True

                # Expr: Math
                elif (i[1] in mathoperator):
                    if (dupeLexeme[index+1] in operands and dupeLexeme[index+2][1]=="Another One Keyword" and dupeLexeme[index+3] in operands):
                        del dupeLexeme[index:index+3]
                        dupeLexeme.insert(index, "expr")
                        flag = True

                # Typecast
                elif (i[1] == "Typecast Declaration"):
                    if(dupeLexeme[index+1]=="varident" and dupeLexeme[index+2]=="Type Literal"):
                        del dupeLexeme[index:index+2]
                        dupeLexeme.insert(index,"typecast")
                        flag = True

                # Comment
                elif (i[1] == "Comment"):
                    del dupeLexeme[index]
                    flag = True

                # Linebreak
                elif (i[1] == "Newline"):
                    del dupeLexeme[index]
                    dupeLexeme.insert(index,"linebreak")
                    flag = True
                elif (i == "linebreak"):
                    if(dupeLexeme[index+1]=="linebreak"):                        
                        del dupeLexeme[index:(index+1)]
                        dupeLexeme.insert(index,"linebreak")
                        flag = True
            if (flag == False):
                break
            
        while(True):
            for i in dupeLexeme:
                flag = False 
                index = dupeLexeme.index(i)
                # Statement
                if (i in ["print", "vardef", "scan"]):
                    del dupeLexeme[index:(index+1)]
                    dupeLexeme.insert(index, "statement")
                    flag = True
                elif (i == "statement"):
                    if(dupeLexeme[index+1]=="linebreak" and dupeLexeme[index+2]=="statement"):
                        del dupeLexeme[index:(index+3)]
                        dupeLexeme.insert(index, "statement")
                        flag = True

            # Program Body
            # if (i[1] == "Code Delimiter"):
            #     if(dupeLexeme[index+1]=="linebreak" and dupeLexeme[index+2]=="statement" and dupeLexeme[index+3]=="linebreak" and dupeLexeme[index+4][1]=="Code Delimiter" and dupeLexeme[index+5]=="statement"):
            #         del dupeLexeme[index:(index+6)]
            #         dupeLexeme.insert(index, "program")
            #         flag = True
            if (flag == False):
                break

        for i in dupeLexeme:
            print(i)
                    
    def get_var_assignments(self, lexemes):
        symbolTable = []

        for i in range(len(lexemes)):
            if lexemes[i][1] == "Variable Declaration":
                var_name = lexemes[i+1][0]  # gets the name of the variable
                    
                if i+1 < len(lexemes) and lexemes[i+1][1] == "Variable Identifier":     # checks if a valid variable name was declared
                    if i+2 < len(lexemes) and lexemes[i+2][1] == "Variable Assignment": # checks if there is a value to be assigned
                        value_token = lexemes[i+3]                                      # gets the value assigned to the variable
                        if lexemes[i+3][1] == "String Delimiter":
                            value_token = lexemes[i+4]
                        
                        if value_token[1] == "Integer Literal":
                            symbolTable.append((var_name, int(value_token[0])))         # if the value assigned is an integer
                        elif value_token[1] == "Float Literal":
                            symbolTable.append((var_name, float(value_token[0])))       # if the value assigned is a float
                        elif value_token[1] == "Boolean Literal" or "String Literal":
                            symbolTable.append((var_name, value_token[0]))              # if the value assigned is of boolean type or string
                    else:
                        symbolTable.append((var_name, None))                            # considers explicit value assignment
                        
        return symbolTable

    def display_symbol(self, symbolTable):
        for i in range(len(symbolTable)):
            token = symbolTable
            self.symbols.insert('', tk.END, values = (token[i][0], token[i][1]))        # displays the values onto the symbol table
            
    
    def getIdentifier(self, lolcode):
        nonIdentifierLexemes = []
        identifierLexemes = []
        validNonIdentifier = list(regexDictionary.keys())[0:-1]
        validIdentifier = list(regexDictionary.keys())[-1]
    
        for current in validNonIdentifier:
            nonIdentifierLexemes = nonIdentifierLexemes + re.findall(current, lolcode)
            lolcode = re.sub(current, " ", lolcode)
        identifierLexemes = re.findall(validIdentifier, lolcode)
        
        identifierLexemes = list(set(identifierLexemes)-set(nonIdentifierLexemes))
        return identifierLexemes

    def getSymbolTable(self, lolcode):
        dupliCode = lolcode
        validRegex = list(regexDictionary.keys())
        identifier = validRegex[-1]
        
        for current in validRegex:
            lexemes = re.findall(current, lolcode)
            lolcode = re.sub(current, " ", lolcode)
            
            for currentLexeme in lexemes:
                symbolTable[currentLexeme] = [regexDictionary[current], None]
                lexemeDictionary[currentLexeme] = current
        
        for current in self.getIdentifier(dupliCode):
            symbolTable[current] = [regexDictionary[identifier], None]
            lexemeDictionary[current] = "Identifier"
            
        return symbolTable
    
    def insertSymbolTable(self, tokens):
        for lexeme in symbolTable.keys():
            self.lexemes.insert('', tk.END, values=(lexeme, symbolTable[lexeme][0]))
            if(symbolTable[lexeme][0] == "Identifier"):
                self.symbols.insert('', tk.END, values=(lexeme, symbolTable[lexeme][1]))
            

root = tk.Tk()
app = InterpreterApp(root)
root.mainloop()
