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
'''
import tkinter as tk
from tkinter import filedialog, ttk, PanedWindow
import re

symbolTable ={}
lexemeDictionary ={}

regexDictionary = {
    r'[\"][^\"]*[\"]': "YARN Literal",
    r'\b-?[0-9]+\b': "NUMBR Literal",
    r'\b-?[0-9]+\.[0-9]+\b': "NUMBAR Literal",
    r'\b(WIN|FAIL)\b': "TROOF Literal",
    r'\b(NUMBR|NUMBAR|YARN|TROOF|NOOB)\b': "TYPE Literal",
    r'\bHAI\b': "Code Delimiter",
    r'\bKTHXBYE\b': "Code Delimiter",
    r'\bWAZZUP\b': "Variable Delimiter",
    r'\bBUHBYE\b': "Variable Delimiter",
    r'\bBTW.*\b': "Comment",
    r'\bOBTW\b': "Multi-line Comment Delimiter",
    r'\bTLDR\b': "Multi-line Comment Delimiter",
    r'\bI HAS A\b': "Variable Declaration",
    r'\bITZ\b': "Variable Assignment",
    r'\bR\b': "R Keyword",
    r'\bSUM OF\b': "SUM OF Keyword",
    r'\bDIFF OF\b': "DIFF OF Keyword",
    r'\bPRODUKT OF\b': "PRODUKT OF Keyword",
    r'\bQUOSHUNT OF\b': "QUOSHUNT OF Keyword",
    r'\bMOD OF\b': "MOD OF Keyword",
    r'\bBIGGR OF\b': "BIGGR OF Keyword",
    r'\bSMALLR OF\b': "SMALLR OF Keyword",
    r'\bBOTH OF\b': "BOTH OF Keyword",
    r'\bEITHER OF\b': "EITHER OF Keyword",
    r'\bWON OF\b': "WON OF Keyword",
    r'\bNOT\b': "NOT Keyword",
    r'\bANY OF\b': "ANY OF Keyword",
    r'\bALL OF\b': "ALL OF Keyword",
    r'\bBOTH SAEM\b': "BOTH SAEM Keyword",
    r'\bDIFFRINT\b': "DIFFRINT Keyword",
    r'\bSMOOSH\b': "SMOOSH Keyword",
    r'\bMAEK\b': "MAEK Keyword",
    r'\bA\b': "A Keyword",
    r'\bIS NOW A\b': "IS NOW A Keyword",
    r'\bVISIBLE\b': "Output Keyword",
    r'\bGIMMEH\b': "GIMMEH Keyword",
    r'\bO RLY\?\b': "O RLY? Keyword",
    r'\bYA RLY\b': "YA RLY Keyword",
    r'\bMEBBE\b': "MEBBE Keyword",
    r'\bNO WAI\b': "NO WAI Keyword",
    r'\bOIC\b': "OIC Keyword",
    r'\bWTF\?\b': "WTF? Keyword",
    r'\bOMG\b': "OMG Keyword",
    r'\bOMGWTF\b': "OMGWTF Keyword",
    r'\bIM IN YR\b': "IM IN YR Keyword",
    r'\bUPPIN\b': "UPPIN Keyword",
    r'\bNERFIN\b': "NERFIN Keyword",
    r'\bYR\b': "YR Keyword",
    r'\bTIL\b': "TIL Keyword",
    r'\bWILE\b': "WILE Keyword",
    r'\bIM OUTTA YR\b': "IM OUTTA YR Keyword",
    r'\bHOW IZ I\b': "HOW IZ I Keyword",
    r'\bIF U SAY SO\b': "IF U SAY SO Keyword",
    r'\bGTFO\b': "GTFO Keyword",
    r'\bFOUND YR\b': "FOUND YR Keyword",
    r'\bI IZ\b': "I IZ Keyword",
    r'\bMKAY\b': "MKAY Keyword",
  
    r'\bAN\b': "AN Keyword",
  
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
        self.text_editor = tk.Text(text_editor_part, height=10, wrap="word") #Allow user to type and edit lol code
        self.text_editor.pack(fill=tk.BOTH, expand=True)
        self.horizontal_pw.add(text_editor_part) #Add to horizontal paned window

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
    # Adds hover effect on buttons
    # -----------------------------------------------------------------------------------------
    def change_on_hover(self, button):
        #change to darker color when arrow is on button
        button.bind("<Enter>", lambda e: button.config(bg="darkgray"))
        
        #go back to usual color of tkinter button when arrow is not on button
        button.bind("<Leave>", lambda e: button.config(bg="SystemButtonFace"))

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

    # -----------------------------------------------------------------------------------------
    # Execute the code from the text editor after pressing the execute button
    #   (3) List of Tokens – This should be updated every time the Execute/Run button (5) is pressed.
    #   (4) Symbol Table – This should be updated every time the Execute/Run button (5) is pressed. 
    # -----------------------------------------------------------------------------------------
    def execute_code(self):
        # Clear previous tokens and symbols
        for item in self.lexemes.get_children():
            self.lexemes.delete(item)
        for item in self.symbols.get_children():
            self.symbols.delete(item)

        # Get the lolcode from the text editor and split into lines (this ensures that the tokens are done per line)
        lolcode = self.text_editor.get("1.0", tk.END).strip().split('\n')

        # Save the current code in the text editor to the actual .lol file
        with open(self.current_file, "w") as file:
            file.write("\n".join(lolcode))

        # 1. Tokenize each line in lolcode and display in the list of tokens (lexemes) - LEXICAL ANALYSIS
        for line in lolcode:
            tokens = self.tokenize_line(line)
            self.insert_tokens(tokens)

    def tokenize_line(self, line):
        tokens = []
        for regex, token_type in regexDictionary.items():
            matches = re.finditer(regex, line)
            for match in matches:
                lexeme = match.group(0)
                if(token_type == "YARN Literal"):
                    tokens.append((lexeme[0], "String Delimiter"))
                    tokens.append((lexeme[1:-1], token_type))
                    tokens.append((lexeme[-1], "String Delimiter"))
                else:
                    tokens.append((lexeme, token_type))
                line = line.replace(lexeme, ' ', 1)  # Replace first occurrence, ensuring a single match of a regex will not match with other ones again
        return tokens

    def insert_tokens(self, tokens):
        for lexeme, token_type in tokens:
            self.lexemes.insert('', tk.END, values=(lexeme, token_type))
            if token_type == "Variable Identifier":
                self.symbols.insert('', tk.END, values=(lexeme, None))

    
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
