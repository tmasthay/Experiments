import re
import sys

def remove_extra(s):
    return '\n'.join([e.strip(' \t') for e in s.split('\n') \
        if len(e.strip(' \t')) > 0])

def remove_latex(text):
    #Step 0: Get body only
    text = re.sub(r"(.*?)\\begin\{document\}(.*?)\\end{document}", r"\2", text, flags=re.DOTALL)

    # Step 1: Remove all LaTeX environments
    text = re.sub(r"\\begin\{.*?\}\n?(.*?)\\end\{.*?\}", r"", text, flags=re.DOTALL)

    # Step 2: Remove all \iffalse environments and their contents
    text = re.sub(r"\\iffalse(.*?)\\fi", "", text, flags=re.DOTALL)

    # Step 3: Remove all lines of the form \iftrue and \fi
    text = re.sub(r"(\\iftrue|\\fi)", "", text)

    # Step 4: Replace all section headers with their contents
    text = re.sub(r"\\(sub)*section\{(.+?)\}", r"\2", text)

    # Step 5: Replace all remaining math expressions with either the plaintext inside or "noun" for non-Latin alphabet words
#    text = re.sub(r"\$([^$]*[^\\])?\$", lambda m: "noun" if re.search(r"[^\x00-\x7F]", m.group(0)) else m.group(0)[1:-1], text)
    text = re.sub(r"\$.*?\$", "noun", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "noun", text, flags=re.DOTALL)

    # Step 6: Replace all other LaTeX commands with the word "placeholder"
    text = re.sub(r"\\cite\{.*?\}", "placeholder", text)
    text = re.sub(r"\\ref\{.*?\}", "placeholder", text)
    text = re.sub(r"\\label\{.*?\}", "placeholder", text)
    text = re.sub(r"\\footnote\{.*?\}", "placeholder", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "placeholder", text)

    return text

if( __name__ == "__main__" ):
    from argparse import *
     
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', default='input.tex', help='Input file path, defaults to input.tex')
    parser.add_argument('--output', default='output.tex', help='Output file path, defaults to output.tex')
    args = parser.parse_args()

    # Open and read the input .tex file
    text = remove_extra(remove_latex(open(args.input,'r').read()))

    # Write the output plaintext to a file
    with open(args.output, 'w') as f:
        f.write(text)

