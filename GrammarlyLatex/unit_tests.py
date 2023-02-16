from omit import remove_latex

def remove_extra(s):
    return '\n'.join([e.strip(' \t') for e in s.split('\n') \
        if len(e.strip(' \t')) > 0])

def test_remove_latex_environments():
    # Tests removal of all latex environments
    input_text = r"""
    This is some text. 
    \begin{equation}
    F=ma
    \end{equation}
    Here is some more text.
    """
    expected_output = input_text.replace(r'\begin{equation}', '') \
        .replace('F=ma','') \
        .replace(r'\end{equation}','')
    
    left = remove_extra(remove_latex(input_text))
    right = remove_extra(expected_output)

    assert left == right

def test_remove_iffalse():
    # Tests removal of all \iffalse environments
    input_text = r"""
    This is some text. 
    \iffalse
    This text should not appear.
    \fi
    Here is some more text.
    """
    expected_output = r"""
    This is some text. 



    Here is some more text.
    """
    left = remove_extra(remove_latex(input_text))
    right = remove_extra(expected_output)
    assert left == right

def test_remove_iftrue():
    # Tests removal of all \iftrue and \fi commands
    input_text = r"""
    This is some text. 
    \iftrue
    This text should appear.
    \fi
    Here is some more text.
    """
    expected_output = r"""
    This is some text. 
     
    This text should appear.

    Here is some more text.
    """
    left = remove_extra(remove_latex(input_text))
    right = remove_extra(expected_output)
    assert left == right

def test_remove_section_headers():
    # Tests removal of all section headers
    input_text = r"""
    \section{Introduction}
    This is the introduction. 
    \subsection{Background}
    This is the background section. 
    """
    expected_output = r"""
    Introduction
    This is the introduction. 
    Background
    This is the background section. 
    """
    assert remove_latex(input_text) == expected_output

def test_replace_math_mode():
    # Tests replacement of math mode with variable names
    input_text = r"""
    This is some text. $F=ma$ Here is some more text. 
    """
    expected_output = input_text.replace(r'$F=ma$','noun')
    assert remove_latex(input_text) == expected_output

def test_remove_misc_latex_commands():
    # Tests removal of miscellaneous latex commands
    input_text = r"""
    This is a citation: \cite{mybook}
    """
    expected_output = r"""
    This is a citation: placeholder
    """
    assert remove_latex(input_text) == expected_output

def test_edge_cases():
    # Tests edge cases (empty string, string with no latex commands)
    input_text = ""
    expected_output = ""
    assert remove_latex(input_text) == expected_output

    input_text = "This is some text."
    expected_output = "This is some text."
    assert remove_latex(input_text) == expected_output

