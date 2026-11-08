"""A parser for the simple programming language

Turns tokens into a syntax tree of `Node` objects.

`Parser` is a recursive descent parser with a built-in operator precedence (Pratt)
parser for expressions. Make your own by subclassing it:

  * Set class attributes for common expression syntax (literals, operators, calls...).
  * Add `@prefix` / `@infix` methods for custom expression syntax.
  * Override `parse_statement` for statements, using helpers like `expect` and
    `parse_block`.

Example:
    class CalcParser(Parser):
        literals = {"NUMBER": parse_number}
        identifier = "IDENT"
        binary_operators = {"+": 10, "-": 10, "*": 20, "/": 20, "^": 30}
        right_associative = {"^"}
        unary_operators = {"-": 25}
        grouping = ("(", ")")
        statement_separator = ";"

        def parse_statement(self):
            if self.match("let"):
                name = self.expect("IDENT")
                self.expect("=")
                value = self.parse_expression()
                self.end_statement()
                return Assign(name.value, value, declare=True, token=name)
            return super().parse_statement()

    tree = CalcParser(Tokenizer("let x = 2 ^ 3; x * -1", RULES)).parse()

Precedence numbers only matter relative to each other: higher binds tighter.
They must be above 0.
"""

import ast
from collections.abc import Callable, Iterable
from typing import Any

from .errors import LanguageError
from .nodes import BinaryOp, Block, Call, Literal, Name, Node, Program, UnaryOp
from .tokenizer import Token

PrefixRule = Callable[["Parser", Token], Node]
InfixRule = Callable[["Parser", Node, Token], Node]


class ParseError(LanguageError):
    """An error related to parsing."""


class UnexpectedTokenError(ParseError):
    """The parser found a token it could not use here."""

    token: Token

    def __init__(self, message: str, token: Token):
        super().__init__(message, token.line, token.column)
        self.token = token


class UnexpectedEOFError(ParseError):
    """The source code ended while the parser still expected more."""


# --- Literal helpers ----------------------------------------------------------


def parse_number(text: str) -> int | float:
    """Converts number text to an int if it has no fraction or exponent, else a float."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def parse_string(text: str) -> str:
    """Converts a quoted string token like `"a\\nb"` to its value, handling escapes.

    Uses Python's string rules, so single or double quotes both work.
    """
    value = ast.literal_eval(text)
    if not isinstance(value, str):
        raise ValueError(f"Not a string literal: {text!r}")
    return value


# --- Tokens -------------------------------------------------------------------


def _end_of(token: Token) -> tuple[int, int]:
    """Returns the line and column just after a token."""
    newlines = token.value.count("\n")
    if newlines:
        return token.line + newlines, len(token.value) - token.value.rfind("\n")
    return token.line, token.column + len(token.value)


def _describe(types: Iterable[str]) -> str:
    return " or ".join(repr(t) for t in types)


class TokenStream:
    """Wraps tokens (a list, or a `Tokenizer` for lazy tokenizing) with lookahead.

    `previous` is the most recently consumed token.
    """

    previous: Token | None

    def __init__(self, tokens: Iterable[Token]):
        self._source = iter(tokens)
        self._buffer: list[Token] = []
        self._exhausted = False
        self.previous = None

    def _fill(self, count: int) -> None:
        while len(self._buffer) < count and not self._exhausted:
            token = next(self._source, None)
            if token is None:
                self._exhausted = True
            else:
                self._buffer.append(token)

    def peek(self, offset: int = 0) -> Token | None:
        """Returns an upcoming token without consuming it, or None past the end."""
        self._fill(offset + 1)
        return self._buffer[offset] if offset < len(self._buffer) else None

    def at_end(self) -> bool:
        """Returns True if there are no tokens left."""
        return self.peek() is None

    def end_position(self) -> tuple[int, int]:
        """Returns the line and column just after the last consumed token."""
        return _end_of(self.previous) if self.previous else (1, 1)

    def advance(self, expected: str = "more input") -> Token:
        """Consumes and returns the next token.

        Raises UnexpectedEOFError (mentioning `expected`) if there are none left.
        """
        token = self.peek()
        if token is None:
            line, column = self.end_position()
            raise UnexpectedEOFError(
                f"Expected {expected} but reached the end of the input.", line, column
            )
        self._buffer.pop(0)
        self.previous = token
        return token

    def check(self, *types: str) -> bool:
        """Returns True if the next token has one of the given types."""
        token = self.peek()
        return token is not None and token.type in types

    def match(self, *types: str) -> Token | None:
        """Consumes and returns the next token if it has one of the given types."""
        return self.advance() if self.check(*types) else None

    def expect(self, *types: str, message: str | None = None) -> Token:
        """Consumes the next token, which must have one of the given types.

        Raises UnexpectedTokenError (or UnexpectedEOFError) otherwise.
        """
        token = self.peek()
        if token is None:
            self.advance(_describe(types))
        assert token is not None
        if token.type not in types:
            raise UnexpectedTokenError(
                message or f"Expected {_describe(types)} but found {token.value!r}.", token
            )
        return self.advance()


# --- Rule decorators ----------------------------------------------------------


def prefix(*token_types: str) -> Callable[[PrefixRule], PrefixRule]:
    """Marks a parser method as the rule for expressions starting with these tokens.

    The method is called as `method(self, token)` with the token already consumed,
    and returns a Node.

        @prefix("if")
        def if_expression(self, token): ...
    """

    def decorate(method: PrefixRule) -> PrefixRule:
        method._langpy_prefix = token_types  # type: ignore[attr-defined]
        return method

    return decorate


def infix(*token_types: str, precedence: int) -> Callable[[InfixRule], InfixRule]:
    """Marks a parser method as the rule for these tokens after an expression.

    The method is called as `method(self, left, token)` with the token already
    consumed, and returns a Node. Use it for operators, indexing, member access, etc.

        @infix("[", precedence=90)
        def index(self, left, token):
            key = self.parse_expression()
            self.expect("]")
            return Index(left, key, token=token)
    """
    if precedence <= 0:
        raise ValueError("Infix precedence must be above 0.")

    def decorate(method: InfixRule) -> InfixRule:
        method._langpy_infix = (token_types, precedence)  # type: ignore[attr-defined]
        return method

    return decorate


# --- Parser -------------------------------------------------------------------


class Parser:
    """Base class for parsers. See the module docs for how to make one.

    Class attributes for common expression syntax (all optional):
        literals: Token type -> converter. A callable is called with the token text
                  (e.g. `parse_number`); anything else is used as the value
                  (e.g. `{"true": True, "nil": None}`). Builds `Literal` nodes.
        identifier: Token type for variable names. Builds `Name` nodes.
        binary_operators: Token type -> precedence. Builds `BinaryOp` nodes.
        right_associative: Binary operators that group right to left (like `^`).
        unary_operators: Token type -> precedence of its operand. Builds `UnaryOp` nodes.
        grouping: (open, close) tokens for parenthesized expressions.
        call: (open, separator, close) tokens for calls like `f(a, b)`. Builds `Call` nodes.
        call_precedence: How tightly calls bind.
        statement_separator: Token that may end statements (like ";"). Optional by
                             default, and repeats are allowed.

    Operator nodes store the token *type* in `op`.
    """

    literals: dict[str, Any] = {}
    identifier: str | None = None
    binary_operators: dict[str, int] = {}
    right_associative: Iterable[str] = frozenset()
    unary_operators: dict[str, int] = {}
    grouping: tuple[str, str] | None = None
    call: tuple[str, str, str] | None = None
    call_precedence: int = 1000
    statement_separator: str | None = None

    _prefix_rules: dict[str, PrefixRule] = {}
    _infix_rules: dict[str, tuple[int, InfixRule]] = {}

    tokens: TokenStream

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._build_rules()

    @classmethod
    def _build_rules(cls) -> None:
        """Collects the class attributes and decorated methods into lookup tables."""
        prefix_rules: dict[str, PrefixRule] = {}
        infix_rules: dict[str, tuple[int, InfixRule]] = {}

        for token_type in cls.literals:
            prefix_rules[token_type] = cls._literal_rule
        if cls.identifier is not None:
            prefix_rules[cls.identifier] = cls._name_rule
        for token_type in cls.unary_operators:
            prefix_rules[token_type] = cls._unary_rule
        if cls.grouping is not None:
            prefix_rules[cls.grouping[0]] = cls._group_rule
        for token_type, precedence in cls.binary_operators.items():
            if precedence <= 0:
                raise ValueError(f"Precedence of {token_type!r} must be above 0.")
            infix_rules[token_type] = (precedence, cls._binary_rule)
        if cls.call is not None:
            infix_rules[cls.call[0]] = (cls.call_precedence, cls._call_rule)

        for name in dir(cls):
            method = getattr(cls, name, None)
            for token_type in getattr(method, "_langpy_prefix", ()):
                prefix_rules[token_type] = method  # type: ignore[assignment]
            infix_info = getattr(method, "_langpy_infix", None)
            if infix_info is not None:
                token_types, precedence = infix_info
                for token_type in token_types:
                    infix_rules[token_type] = (precedence, method)  # type: ignore[assignment]

        cls._prefix_rules = prefix_rules
        cls._infix_rules = infix_rules

    def __init__(self, tokens: Iterable[Token] | TokenStream):
        self.tokens = tokens if isinstance(tokens, TokenStream) else TokenStream(tokens)

    # --- Token helpers (shortcuts for self.tokens) ---

    def peek(self, offset: int = 0) -> Token | None:
        """Returns an upcoming token without consuming it, or None past the end."""
        return self.tokens.peek(offset)

    def advance(self, expected: str = "more input") -> Token:
        """Consumes and returns the next token."""
        return self.tokens.advance(expected)

    def check(self, *types: str) -> bool:
        """Returns True if the next token has one of the given types."""
        return self.tokens.check(*types)

    def match(self, *types: str) -> Token | None:
        """Consumes and returns the next token if it has one of the given types."""
        return self.tokens.match(*types)

    def expect(self, *types: str, message: str | None = None) -> Token:
        """Consumes the next token, which must have one of the given types."""
        return self.tokens.expect(*types, message=message)

    def at_end(self) -> bool:
        """Returns True if there are no tokens left."""
        return self.tokens.at_end()

    def error(self, message: str, token: Token | None = None) -> ParseError:
        """Creates a ParseError at `token` (default: the next token) for you to raise."""
        token = token or self.peek()
        if token is None:
            return UnexpectedEOFError(message, *self.tokens.end_position())
        return UnexpectedTokenError(message, token)

    # --- Program structure ---

    def parse(self) -> Node:
        """Parses the whole input. By default returns a `Program` of statements."""
        statements = self.parse_statements()
        token = self.peek()
        if token is not None:
            raise UnexpectedTokenError(f"Unexpected {token.value!r}.", token)
        return Program(statements)

    def parse_statement(self) -> Node:
        """Parses one statement. By default an expression followed by `end_statement`.

        Override this to add statements like `if`, `while` or `let`.
        """
        expression = self.parse_expression()
        self.end_statement()
        return expression

    def end_statement(self) -> None:
        """Consumes any statement separators (if `statement_separator` is set)."""
        if self.statement_separator is not None:
            while self.match(self.statement_separator):
                pass

    def parse_statements(self, until: str | None = None) -> list[Node]:
        """Parses statements until a token of type `until` (not consumed) or the end."""
        statements: list[Node] = []
        while True:
            self.end_statement()
            if self.at_end() or (until is not None and self.check(until)):
                return statements
            statements.append(self.parse_statement())

    def parse_block(self, open: str = "{", close: str = "}") -> Block:
        """Parses `open statements... close` into a Block."""
        start = self.expect(open)
        statements = self.parse_statements(until=close)
        self.expect(close)
        return Block(statements, token=start)

    def parse_delimited(
        self,
        close: str,
        separator: str = ",",
        item: Callable[[], Any] | None = None,
        open: str | None = None,
    ) -> list[Any]:
        """Parses a separated list like `a, b, c)` and consumes the closing token.

        open: If given, it is expected first. Leave it out if already consumed.
        item: Parses one item. Defaults to `parse_expression`.
        A trailing separator is allowed.
        """
        if open is not None:
            self.expect(open)
        parse_item = item or self.parse_expression
        items: list[Any] = []
        while not self.match(close):
            items.append(parse_item())
            if not self.match(separator):
                self.expect(close)
                break
        return items

    # --- Expressions ---

    def parse_expression(self, precedence: int = 0) -> Node:
        """Parses an expression whose operators bind tighter than `precedence`."""
        token = self.advance("an expression")
        rule = self._prefix_rules.get(token.type)
        if rule is None:
            raise UnexpectedTokenError(f"Expected an expression but found {token.value!r}.", token)
        left = rule(self, token)

        while True:
            token = self.peek()
            if token is None:
                return left
            entry = self._infix_rules.get(token.type)
            if entry is None or entry[0] <= precedence:
                return left
            self.advance()
            left = entry[1](self, left, token)

    def precedence_of(self, token_type: str) -> int:
        """Returns the infix precedence of a token type, or 0 if it isn't infix."""
        entry = self._infix_rules.get(token_type)
        return entry[0] if entry else 0

    # --- Built-in rules for the class attributes (override to build other nodes) ---

    def _literal_rule(self, token: Token) -> Node:
        converter = self.literals[token.type]
        if not callable(converter):
            return Literal(converter, token=token)
        try:
            return Literal(converter(token.value), token=token)
        except (ValueError, SyntaxError) as error:
            raise UnexpectedTokenError(f"Invalid literal {token.value!r}.", token) from error

    def _name_rule(self, token: Token) -> Node:
        return Name(token.value, token=token)

    def _unary_rule(self, token: Token) -> Node:
        operand = self.parse_expression(self.unary_operators[token.type])
        return UnaryOp(token.type, operand, token=token)

    def _group_rule(self, token: Token) -> Node:
        assert self.grouping is not None
        expression = self.parse_expression()
        self.expect(self.grouping[1])
        return expression

    def _binary_rule(self, left: Node, token: Token) -> Node:
        precedence = self.binary_operators[token.type]
        if token.type in self.right_associative:
            precedence -= 1
        right = self.parse_expression(precedence)
        return BinaryOp(token.type, left, right, token=token)

    def _call_rule(self, left: Node, token: Token) -> Node:
        assert self.call is not None
        _, separator, close = self.call
        args = self.parse_delimited(close, separator)
        return Call(left, args, token=token)


Parser._build_rules()
