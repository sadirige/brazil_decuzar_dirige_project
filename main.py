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
'''
import tkinter as tk
from tkinter import filedialog, ttk, PanedWindow

class InterpreterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Team (ﾉ◕ヮ◕)ﾉ*:･ﾟ LOLTERPRETER")
        self.create_widgets()
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
        text_editor = tk.Text(text_editor_part, height=10, wrap="word") #Allow user to type and edit lol code
        text_editor.pack(fill=tk.BOTH, expand=True)
        self.horizontal_pw.add(text_editor_part) #Add to horizontal paned window

        #(3) List of Tokens - This should be updated every time the Execute/Run button (5) is pressed. This should contain all the lexemes detected from the code being ran, and their classification.
        tokens_list_part = tk.Frame(self.horizontal_pw)
        tokens_label = tk.Label(tokens_list_part, text="Lexemes")
        tokens_label.pack(anchor="center")
        lexemes = ttk.Treeview(tokens_list_part, columns=("Lexeme", "Classification"), show="headings") #Divide area into 2 columns
        lexemes.heading("Lexeme", text="Lexeme", anchor="w")
        lexemes.heading("Classification", text="Classification", anchor="w")
        lexemes.pack(fill=tk.BOTH, expand=True)
        self.horizontal_pw.add(tokens_list_part) #Add to horizontal paned window

        #(4) Symbol Table - This should be updated every time the Execute/Run button (5) is pressed. This should contain all the variables available in the program being ran, and their updated values.
        symbol_table_part = tk.Frame(self.horizontal_pw)
        symbol_label = tk.Label(symbol_table_part, text="SYMBOL TABLE")
        symbol_label.pack(anchor="center")
        symbols = ttk.Treeview(symbol_table_part, columns=("Identifier", "Value"), show="headings") #Also divide area into 2 columns
        symbols.heading("Identifier", text="Identifier", anchor="w")
        symbols.heading("Value", text="Value", anchor="w")
        symbols.pack(fill=tk.BOTH, expand=True)
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
        self.root.update_idletasks()
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
            file_name = file_path.split("/")[-1]
            self.file_name.config(text=file_name) #Update (None) to the selected file name
        else:
            tk.messagebox.showerror("Invalid File", "Please select a valid .lol file.")

        #TODO: Load the contents of the selected file into the text editor

    # -----------------------------------------------------------------------------------------
    # Execute the code from the text editor after pressing the execute button
    # -----------------------------------------------------------------------------------------
    def execute_code(self):
        pass


# Create and start the app in maximized state
root = tk.Tk()
root.state("zoomed")
app = InterpreterApp(root)
root.mainloop()
