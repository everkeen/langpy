"""Mini: a small scripting language built with langpy

Run a file or start a REPL:
    python -m langpy.examples.mini [file.mini]

Example program:
    # comments start with #
    let greeting = "hello";

    fn fact(n) {
        if n <= 1 { return 1; }
        return n * fact(n - 1);
    }

    let i = 0;
    while i < 5 {
        i = i + 1;
        if i == 2 { continue; }
        print(i, fact(i));
    }

    let double = fn(x) x * 2;     # anonymous function with an expression body
    print(double(21), not false and true);
"""

from langpy import (
    Assign,
    Break,
    Continue,
    FunctionDef,
    If,
    Language,
    Parser,
    Return,
    While,
    parse_number,
    parse_string,
    prefix,
)
from langpy import ValidToken as T

KEYWORDS = ["let", "fn", "if", "else", "while", "return", "break", "continue",
            "true", "false", "nil", "and", "or", "not"]  # fmt: skip
SYMBOLS = ["+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "=",
           "(", ")", "{", "}", ",", ";"]  # fmt: skip

TOKENS = [
    *(T.exact(word) for word in KEYWORDS),
    *(T.exact(symbol) for symbol in SYMBOLS),
    T.regex(r"[A-Za-z_][A-Za-z0-9_]*", name="IDENT"),
    T.regex(r"[0-9]+(\.[0-9]+)?", name="NUMBER"),
    T.regex(r'"(\\.|[^"\\\n])*"', name="STRING"),
    T.regex(r"#[^\n]*", skip=True),
]


class MiniParser(Parser):
    literals = {
        "NUMBER": parse_number,
        "STRING": parse_string,
        "true": True,
        "false": False,
        "nil": None,
    }
    identifier = "IDENT"
    binary_operators = {
        "or": 10,
        "and": 20,
        "==": 30, "!=": 30,
        "<": 40, "<=": 40, ">": 40, ">=": 40,
        "+": 50, "-": 50,
        "*": 60, "/": 60, "%": 60,
    }  # fmt: skip
    unary_operators = {"-": 70, "not": 25}
    grouping = ("(", ")")
    call = ("(", ",", ")")
    statement_separator = ";"

    def parse_statement(self):
        token = self.peek()
        if self.match("let"):
            name = self.expect("IDENT")
            self.expect("=")
            value = self.parse_expression()
            self.end_statement()
            return Assign(name.value, value, declare=True, token=token)
        if self.check("IDENT") and self.peek(1) and self.peek(1).type == "=":
            name = self.advance()
            self.advance()
            value = self.parse_expression()
            self.end_statement()
            return Assign(name.value, value, token=name)
        if self.match("if"):
            return self.parse_if(token)
        if self.match("while"):
            condition = self.parse_expression()
            return While(condition, self.parse_block(), token=token)
        if self.check("fn") and self.peek(1) and self.peek(1).type == "IDENT":
            self.advance()
            name = self.advance()
            return self.parse_function_rest(name.value, token)
        if self.match("return"):
            value = None
            if not self.at_end() and not self.check(";", "}"):
                value = self.parse_expression()
            self.end_statement()
            return Return(value, token=token)
        if self.match("break"):
            self.end_statement()
            return Break(token=token)
        if self.match("continue"):
            self.end_statement()
            return Continue(token=token)
        return super().parse_statement()

    def parse_if(self, token):
        condition = self.parse_expression()
        then = self.parse_block()
        otherwise = None
        if self.match("else"):
            else_if = self.match("if")
            otherwise = self.parse_if(else_if) if else_if else self.parse_block()
        return If(condition, then, otherwise, token=token)

    def parse_function_rest(self, name, token):
        params = self.parse_delimited(")", item=lambda: self.expect("IDENT").value, open="(")
        body = self.parse_block() if self.check("{") else self.parse_expression()
        return FunctionDef(name, params, body, token=token)

    @prefix("fn")
    def anonymous_function(self, token):
        return self.parse_function_rest(None, token)


class Mini(Language):
    name = "mini"
    tokens = TOKENS
    parser_class = MiniParser


if __name__ == "__main__":
    raise SystemExit(Mini().main())
