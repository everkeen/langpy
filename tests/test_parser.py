"""Tests for the parser."""

import pytest

from langpy import (
    Assign,
    BinaryOp,
    Block,
    Call,
    Literal,
    Name,
    Node,
    ParseError,
    Parser,
    Program,
    Token,
    TokenStream,
    Tokenizer,
    UnaryOp,
    UnexpectedEOFError,
    UnexpectedTokenError,
    infix,
    parse_number,
    parse_string,
    prefix,
)
from langpy import ValidToken as T

RULES = [
    *(T.exact(s) for s in ["+", "-", "*", "/", "^", "(", ")", "[", "]", ",", ";", "{", "}", "="]),
    *(T.exact(k) for k in ["true", "nil", "let", "not"]),
    T.regex(r"[A-Za-z_]\w*", name="IDENT"),
    T.regex(r"[0-9]+(\.[0-9]+)?", name="NUMBER"),
    T.regex(r'"(\\.|[^"\\])*"', name="STRING"),
]


class ExprParser(Parser):
    literals = {"NUMBER": parse_number, "STRING": parse_string, "true": True, "nil": None}
    identifier = "IDENT"
    binary_operators = {"+": 10, "-": 10, "*": 20, "/": 20, "^": 30}
    right_associative = {"^"}
    unary_operators = {"-": 25, "not": 5}
    grouping = ("(", ")")
    call = ("(", ",", ")")
    statement_separator = ";"


def parse(source: str, parser: type[Parser] = ExprParser) -> Node:
    return parser(Tokenizer(source, RULES)).parse()


def expr(source: str, parser: type[Parser] = ExprParser) -> Node:
    program = parse(source, parser)
    assert isinstance(program, Program) and len(program.statements) == 1
    return program.statements[0]


def num(value):
    return Literal(value)


# --- Literal helpers ----------------------------------------------------------


def test_parse_number():
    assert parse_number("12") == 12 and isinstance(parse_number("12"), int)
    assert parse_number("1.5") == 1.5


def test_parse_string():
    assert parse_string(r'"a\n\"b\""') == 'a\n"b"'
    assert parse_string("'single'") == "single"
    with pytest.raises(ValueError):
        parse_string("12")


# --- Expressions --------------------------------------------------------------


def test_literals():
    assert expr("42") == num(42)
    assert expr('"hi"') == num("hi")
    assert expr("true") == num(True)
    assert expr("nil") == num(None)


def test_name():
    assert expr("abc") == Name("abc")


def test_precedence():
    assert expr("1 + 2 * 3") == BinaryOp("+", num(1), BinaryOp("*", num(2), num(3)))
    assert expr("1 * 2 + 3") == BinaryOp("+", BinaryOp("*", num(1), num(2)), num(3))


def test_left_associative():
    assert expr("1 - 2 - 3") == BinaryOp("-", BinaryOp("-", num(1), num(2)), num(3))


def test_right_associative():
    assert expr("2 ^ 3 ^ 4") == BinaryOp("^", num(2), BinaryOp("^", num(3), num(4)))


def test_grouping():
    assert expr("(1 + 2) * 3") == BinaryOp("*", BinaryOp("+", num(1), num(2)), num(3))


def test_unary():
    assert expr("-1 * 2") == BinaryOp("*", UnaryOp("-", num(1)), num(2))
    assert expr("- -x") == UnaryOp("-", UnaryOp("-", Name("x")))
    # A low-precedence unary takes the whole expression after it.
    assert expr("not 1 + 2") == UnaryOp("not", BinaryOp("+", num(1), num(2)))


def test_calls():
    assert expr("f()") == Call(Name("f"), [])
    assert expr("f(1, x + 1)") == Call(Name("f"), [num(1), BinaryOp("+", Name("x"), num(1))])
    assert expr("f(1,)") == Call(Name("f"), [num(1)])
    assert expr("f(1)(2)") == Call(Call(Name("f"), [num(1)]), [num(2)])
    assert expr("-f(1)") == UnaryOp("-", Call(Name("f"), [num(1)]))


def test_nodes_keep_their_token():
    node = expr("a +\n  b")
    assert isinstance(node, BinaryOp)
    assert (node.line, node.column) == (1, 3)
    assert (node.right.line, node.right.column) == (2, 3)


def test_statements_and_separators():
    program = parse(";1;; 2\n3;")
    assert program == Program([num(1), num(2), num(3)])


def test_empty_program():
    assert parse("") == Program([])


# --- Errors -------------------------------------------------------------------


def test_missing_operand():
    with pytest.raises(UnexpectedEOFError) as err:
        parse("1 +")
    assert (err.value.line, err.value.column) == (1, 4)


def test_unexpected_token():
    with pytest.raises(UnexpectedTokenError) as err:
        parse("1 + )")
    assert err.value.token.value == ")"
    assert (err.value.line, err.value.column) == (1, 5)


def test_unclosed_group():
    with pytest.raises(UnexpectedEOFError):
        parse("(1 + 2")


def test_missing_comma():
    with pytest.raises(UnexpectedTokenError) as err:
        parse("f(1 2)")
    assert "Expected ')'" in str(err.value)


def test_error_hierarchy():
    with pytest.raises(ParseError):
        parse("*")


def test_bad_precedence_is_rejected():
    with pytest.raises(ValueError):

        class Bad(Parser):
            binary_operators = {"+": 0}

    with pytest.raises(ValueError):
        infix("+", precedence=0)


# --- Customizing --------------------------------------------------------------


class CustomParser(ExprParser):
    def parse_statement(self):
        if self.match("let"):
            name = self.expect("IDENT")
            self.expect("=")
            value = self.parse_expression()
            self.end_statement()
            return Assign(name.value, value, declare=True, token=name)
        if self.check("{"):
            return self.parse_block()
        return super().parse_statement()

    @prefix("[")
    def list_literal(self, token):
        return Call(Name("list"), self.parse_delimited("]"), token=token)

    @infix("[", precedence=100)
    def index(self, left, token):
        key = self.parse_expression()
        self.expect("]")
        return Call(Name("index"), [left, key], token=token)


def test_custom_statement():
    assert expr("let x = 1 + 2;", CustomParser) == Assign(
        "x", BinaryOp("+", num(1), num(2)), declare=True
    )


def test_custom_statement_error():
    with pytest.raises(UnexpectedTokenError) as err:
        parse("let 5 = 1", CustomParser)
    assert "Expected 'IDENT'" in str(err.value)


def test_block():
    assert expr("{ 1; { 2 } }", CustomParser) == Block([num(1), Block([num(2)])])


def test_unclosed_block():
    with pytest.raises(UnexpectedEOFError):
        parse("{ 1;", CustomParser)


def test_prefix_and_infix_decorators():
    assert expr("[1, 2][0]", CustomParser) == Call(
        Name("index"), [Call(Name("list"), [num(1), num(2)]), num(0)]
    )


def test_decorated_rules_are_inherited_and_independent():
    class Child(CustomParser):
        pass

    assert expr("[1]", Child) == Call(Name("list"), [num(1)])
    with pytest.raises(ParseError):
        parse("[1]")  # the base ExprParser doesn't know "["


def test_overriding_a_built_in_rule():
    class Folding(ExprParser):
        def _binary_rule(self, left, token):
            node = super()._binary_rule(left, token)
            if isinstance(node.left, Literal) and isinstance(node.right, Literal):
                if node.op == "+":
                    return Literal(node.left.value + node.right.value, token=token)
            return node

    assert expr("1 + 2 + 3", Folding) == num(6)


def test_parser_without_rules_cannot_parse_expressions():
    with pytest.raises(UnexpectedTokenError):
        parse("1", Parser)


def test_error_helper():
    class Strict(ExprParser):
        def parse_statement(self):
            if self.check("NUMBER"):
                raise self.error("Numbers alone are not allowed.")
            return super().parse_statement()

    with pytest.raises(UnexpectedTokenError, match="Numbers alone"):
        parse("x; 5", Strict)
    with pytest.raises(UnexpectedEOFError):
        raise Strict([]).error("at end")


# --- TokenStream --------------------------------------------------------------


def make_token(type_: str, value: str, column: int) -> Token:
    return Token(type_, value, 1, column, column - 1)


def test_token_stream_from_list():
    stream = TokenStream([make_token("A", "a", 1), make_token("B", "bb", 3)])
    assert stream.peek(1).type == "B"
    assert stream.match("B") is None
    assert stream.expect("A").value == "a"
    assert stream.previous.value == "a"
    assert stream.check("A", "B")
    stream.advance()
    assert stream.at_end()
    assert stream.end_position() == (1, 5)
    with pytest.raises(UnexpectedEOFError):
        stream.advance()


def test_token_stream_is_lazy():
    consumed = []

    def tokens():
        for i in range(3):
            consumed.append(i)
            yield make_token("N", str(i), i + 1)

    stream = TokenStream(tokens())
    stream.peek()
    assert consumed == [0]
    stream.peek(1)
    assert consumed == [0, 1]
