'''
This part is the main file where the GUI is loaded for the Interpreter
'''
import tkinter as tk
from tkinter import filedialog, ttk, PanedWindow
from lexer import tokenize
from parser import Parser
from semantic_analyzer import SemanticAnalyzer

class InterpreterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Team (ﾉ◕ヮ◕)ﾉ*:･ﾟ LOLTERPRETER")
        self.current_file = None
        self.create_widgets()

        #Initialize window in maximized state
        self.root.update_idletasks()
        try:
            self.root.wm_attributes('-zoomed', True) #Works for Ubuntu and others, not Arch
        except:
            self.root.state('zoomed') #Works for Windows and others, not Ubuntu

        self.root.after(100, self.initialize_parts_sashpos)  #Update sash position after main loop starts

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
        self.line_numbers.pack(side="left", fill="y") #Add line numbers beside each line of lolcode on text editor
        editor_scrollbar = tk.Frame(text_editor_part)
        editor_scrollbar.pack(side="right", fill="both", expand=True) #Create a frame for text editor and a scrollbar on its right
        self.text_editor = tk.Text(text_editor_part, wrap="none", undo=True)
        self.text_editor.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(editor_scrollbar, command=self.sync_scroll) 
        scrollbar.pack(side="right", fill="y")
        self.text_editor.configure(yscrollcommand=scrollbar.set)

        #Synchronize vertical scrolling between line numbers and text editor, meaning if you scroll one, the other will scroll too
        self.line_numbers.bind("<MouseWheel>", self.on_mouse_scroll) 
        self.text_editor.bind("<MouseWheel>", self.on_mouse_scroll)

        #Define horizontal scrolling since its functionality is removed once we customized the vertical scrolling
        self.text_editor.bind("<Shift-MouseWheel>", self.on_horizontal_scroll)

        #Bind events to update line numbers counting the lines in the text editor
        self.text_editor.bind("<KeyRelease>", self.update_line_numbers)
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

        #Now we create the lower part of the vertical paned window (contains execute button and console)
        execute_console_part_frame = tk.Frame(self.vertical_pw)
        
        #(5) Execute/Run button - This will run the code from the text editor (2).
        execute_button = tk.Button(execute_console_part_frame, text="EXECUTE", command=self.execute_code)
        execute_button.pack(fill=tk.X, pady=(0, 5))
        self.change_on_hover(execute_button) #Add hover effect on button

        #(6) Console - Input/Output of the program should be reflected in the console. For variable input, you can add a separate field for user input, or have a dialog box pop up.
        self.console_part = tk.Text(execute_console_part_frame, height=5, wrap="word")
        self.console_part.pack(fill=tk.BOTH, expand=True)
        self.console_part.config(state="disabled")

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
    # Synchronizes vertical scrolling between line numbers and text editor using the scrollbar widget
    # -----------------------------------------------------------------------------------------
    def sync_scroll(self, *args):
        self.line_numbers.yview(*args)
        self.text_editor.yview(*args)

    # -----------------------------------------------------------------------------------------
    # Synchronizes vertical scrolling between line numbers and text editor using the mouse wheel
    # -----------------------------------------------------------------------------------------
    def on_mouse_scroll(self, event):
        if event.widget == self.line_numbers:
            delta = -1 if event.delta > 0 else 1
            self.line_numbers.yview_scroll(delta, "units")
            self.text_editor.yview_moveto(self.line_numbers.yview()[0])
        elif event.widget == self.text_editor:
            delta = -1 if event.delta > 0 else 1
            self.text_editor.yview_scroll(delta, "units")
            self.line_numbers.yview_moveto(self.text_editor.yview()[0])
        return "break"

    # -----------------------------------------------------------------------------------------
    # Handles horizontal scrolling in the text editor (has to be defined since vertical scolling was defined)
    # -----------------------------------------------------------------------------------------
    def on_horizontal_scroll(self, event):
        delta = -1 if event.delta > 0 else 1
        self.text_editor.xview_scroll(delta, "units")
        return "break" 

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
        #Clear previous lexemes, symbols, and console output
        for item in self.lexemes.get_children():
            self.lexemes.delete(item)
        for item in self.symbols.get_children():
            self.symbols.delete(item)
        self.console_part.config(state="normal")
        self.console_part.delete(1.0, tk.END)

        #Get a copy of the lolcode from the text editor
        lolcode = self.text_editor.get("1.0", tk.END).splitlines()

        #Save the current code in the text editor to the actual .lol file
        if(self.current_file is not None):
            with open(self.current_file, "w") as file:
                file.write("\n".join(lolcode))

        #1. Tokenize each line in lolcode and display in the list of tokens (lexemes) - LEXICAL ANALYSIS
        lexemes = tokenize(lolcode)

        # for i in lexemes:
        #     print(i)
        print(lexemes)

        self.display_lexemes(lexemes)

        #2. Convert tokens to symbol table - SYNTAX ANALYSIS
        parse = Parser(lexemes, self.console_part)
        symbol_table, ast = parse.program()
        self.display_symbol_table(symbol_table)

        if ast[0] == "Syntax Error":
            #do something else
            print("failed")
            self.console_part.insert(tk.END, "Syntax Error: " + ast[1])

        #3. If syntax is correct (there is a Generated AST from syntax analysis), perform semantic analysis
        else:
            #semantic analysis
            # print("success")
            print(ast)
            semantic = SemanticAnalyzer(symbol_table, ast[1], self.console_part, self.symbols)
            semantic.run()

    # -----------------------------------------------------------------------------------------
    # Updates the Lexemes part of the GUI once the lines of the lolcode in the text editor are tokenized
    # -----------------------------------------------------------------------------------------
    def display_lexemes(self, lexemes):
        for lexeme, classification, line_number in lexemes:
            if lexeme == "\n" and classification == "Linebreak":
                continue
            elif lexeme == "BTW" or lexeme == "OBTW" or lexeme == "TLDR":
                continue
            elif classification == "Lexical Error":
                error_message = "Lexical Error: Unrecognized token: " + lexeme + " at line " + str(line_number) + ".\n"
                self.console_part.insert(tk.END, error_message)
            else:
                #ignore newline and comments, only show the other more important lexemes
                self.lexemes.insert('', tk.END, values=(lexeme, classification))

    # -----------------------------------------------------------------------------------------
    # Updates the Symbol Table part of the GUI once the tokens' syntax are checked
    # -----------------------------------------------------------------------------------------
    def display_symbol_table(self, symbol_table):
        for variable, (_, value) in symbol_table.items():
            self.symbols.insert('', tk.END, values=(variable, value))

    def display_console(self, output):
        for line in output:
            self.console_part.insert(tk.END, line)
            

# Create and run the app
root = tk.Tk()
app = InterpreterApp(root)
root.mainloop()
