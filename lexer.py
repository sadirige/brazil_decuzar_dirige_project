import re

symbolTable = {}
lexemeDictionary = {}

'''
    Regex for lexeme matching paired with its type 
'''
regexDictionary = {
    r'^-?[0-9]+$': "NUMBR Literal",
    r'^-?[0-9]+\.[0-9]+$': "NUMBAR Literal",
    r'^"([^"\\]|\\.)*"$': "YARN Literal",
    r'^(WIN|FAIL)$': "TROOF Literal",
    r'^(NUMBR|NUMBAR|YARN|TROOF)$': "TYPE Literal",

    r'\bHAI\b': "HAI Keyword",
    r'\bKTHXBYE\b': "KTHXBYE Keyword",

    r'^WAZZUP$': "WAZZUP Keyword",
    r'^BUHBYE$': "BUHBYE Keyword",
    r'^BTW$': "BTW Keyword",
    r'^OBTW$': "OBTW Keyword",
    r'^TLDR$': "TLDR Keyword",
    r'^I HAS A$': "I HAS A Keyword",
    r'^ITZ$': "ITZ Keyword",
    r'^R$': "R Keyword",
    r'^SUM OF$': "SUM OF Keyword",
    r'^DIFF OF$': "DIFF OF Keyword",
    r'^PRODUKT OF$': "PRODUKT OF Keyword",
    r'^QUOSHUNT OF$': "QUOSHUNT OF Keyword",
    r'^MOD OF$': "MOD OF Keyword",
    r'^BIGGR OF$': "BIGGR OF Keyword",
    r'^SMALLR OF$': "SMALLR OF Keyword",
    r'^BOTH OF$': "BOTH OF Keyword",
    r'^EITHER OF$': "EITHER OF Keyword",
    r'^WON OF$': "WON OF Keyword",
    r'^NOT$': "NOT Keyword",
    r'^ANY OF$': "ANY OF Keyword",
    r'^ALL OF$': "ALL OF Keyword",
    r'^BOTH SAEM$': "BOTH SAEM Keyword",
    r'^DIFFRINT$': "DIFFRINT Keyword",
    r'^SMOOSH$': "SMOOSH Keyword",
    r'^MAEK$': "MAEK Keyword",
    r'^A$': "A Keyword",
    r'^IS NOW A$': "IS NOW A Keyword",
    r'^VISIBLE$': "VISIBLE Keyword",
    r'^GIMMEH$': "GIMMEH Keyword",
    r'^O RLY\?$': "O RLY? Keyword",
    r'^YA RLY$': "YA RLY Keyword",
    r'^MEBBE$': "MEBBE Keyword",
    r'^NO WAI$': "NO WAI Keyword",
    r'^OIC$': "OIC Keyword",
    r'^WTF\?$': "WTF? Keyword",
    r'^OMG$': "OMG Keyword",
    r'^OMGWTF$': "OMGWTF Keyword",
    r'^IM IN YR$': "IM IN YR Keyword",
    r'^UPPIN$': "UPPIN Keyword",
    r'^NERFIN$': "NERFIN Keyword",
    r'^YR$': "YR Keyword",
    r'^TIL$': "TIL Keyword",
    r'^WILE$': "WILE Keyword",
    r'^IM OUTTA YR$': "IM OUTTA YR Keyword",
    r'^HOW IZ I$': "HOW IZ I Keyword",
    r'^IF U SAY SO$': "IF U SAY SO Keyword",
    r'^GTFO$': "GTFO Keyword",
    r'^FOUND YR$': "FOUND YR Keyword",
    r'^I IZ$': "I IZ Keyword",
    r'^MKAY$': "MKAY Keyword",
    r'^[a-zA-Z][a-zA-Z0-9_]*$': "Identifier"
}

def readlolcode(lolcodeFile):
    with open(lolcodeFile) as file:
        code = "".join(file.readlines())
    return code

def getSymbolTable(lolcode):
    validRegex = list(regexDictionary.keys())

    for current in validRegex:
        # find the match
        lexemes = re.findall(current, lolcode)
        # remove the matched in the code
        code = re.sub(current, " ", lolcode)

        # add matched in symbol table
        for currentLexeme in lexemes:
            symbolTable[currentLexeme] = [regexDictionary[current], None]
            lexemeDictionary[currentLexeme] = current
    
    return symbolTable

def printTable(symbolTable):
    print("LEXEME-TYPE-VALUE")
    for lexeme in symbolTable.keys():
        print(lexeme,"-", symbolTable[lexeme][0], "-", symbolTable[lexeme][1])

def main():
    lolcode = readlolcode("sample.lol")
    tokens = getSymbolTable(lolcode)
    printTable(tokens)
    
if __name__ == "__main__":
    main()