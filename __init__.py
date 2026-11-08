"""A simple programming language-making system for python

Makes it a little easier to create and use custom programming languages.

A language is made of three parts, each a class you can subclass:
    Tokenizer   - splits source code into tokens, using `ValidToken` rules.
    Parser      - turns tokens into a tree of `Node`s.
    Interpreter - runs the tree.
`Language` ties them together and adds `run`, `run_file`, a REPL and a command line.

See `langpy.examples.mini` for a complete small language.
"""

from .errors import LanguageError
from .interpreter import (
    Environment,
    Function,
    Interpreter,
    LanguageRuntimeError,
    UndefinedVariableError,
)
from .language import Language
from .nodes import (
    Assign,
    BinaryOp,
    Block,
    Break,
    Call,
    Continue,
    FunctionDef,
    If,
    Literal,
    Name,
    Node,
    NodeVisitor,
    Program,
    Return,
    UnaryOp,
    While,
)
from .parser import (
    ParseError,
    Parser,
    TokenStream,
    UnexpectedEOFError,
    UnexpectedTokenError,
    infix,
    parse_number,
    parse_string,
    prefix,
)
from .tokenizer import (
    InvalidTokenError,
    ReachedEOFError,
    Token,
    Tokenizer,
    TokenizerError,
    ValidToken,
    ValidTokenRule,
)

__all__ = [
    # errors
    "LanguageError",
    "TokenizerError",
    "InvalidTokenError",
    "ReachedEOFError",
    "ParseError",
    "UnexpectedTokenError",
    "UnexpectedEOFError",
    "LanguageRuntimeError",
    "UndefinedVariableError",
    # tokenizer
    "Token",
    "Tokenizer",
    "ValidToken",
    "ValidTokenRule",
    # parser
    "Parser",
    "TokenStream",
    "prefix",
    "infix",
    "parse_number",
    "parse_string",
    # nodes
    "Node",
    "NodeVisitor",
    "Program",
    "Block",
    "Literal",
    "Name",
    "UnaryOp",
    "BinaryOp",
    "Call",
    "Assign",
    "If",
    "While",
    "Break",
    "Continue",
    "FunctionDef",
    "Return",
    # interpreter
    "Interpreter",
    "Environment",
    "Function",
    # language
    "Language",
]
