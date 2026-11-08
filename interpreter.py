"""An interpreter for the simple programming language

Runs a syntax tree by visiting each node. The base `Interpreter` already knows how
to run the ready-made nodes in `nodes` (literals, variables, operators, calls, if,
while, functions...). Subclass it to change behaviour or to run your own nodes:

    class MyInterpreter(Interpreter):
        binary_operators = {**Interpreter.binary_operators, "..": lambda a, b: str(a) + str(b)}
        builtins = {**Interpreter.builtins, "sqrt": math.sqrt}

        def visit_Print(self, node):
            print(self.visit(node.value))

Any Python callable can be stored in a variable and called from the language.
"""

import operator
from collections.abc import Callable, Iterator
from typing import Any

from .errors import LanguageError
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


class LanguageRuntimeError(LanguageError):
    """An error that happened while running a program."""


class UndefinedVariableError(LanguageRuntimeError):
    """A variable was used before it was defined."""

    name: str

    def __init__(self, name: str, line: int | None = None, column: int | None = None):
        super().__init__(f"Variable {name!r} is not defined.", line, column)
        self.name = name


# Control flow signals. They are BaseExceptions so that `except Exception` in
# user code or builtins never swallows them.


class _ControlFlow(BaseException):
    node: Node

    def __init__(self, node: Node):
        super().__init__()
        self.node = node


class _ReturnSignal(_ControlFlow):
    def __init__(self, node: Node, value: Any):
        super().__init__(node)
        self.value = value


class _BreakSignal(_ControlFlow):
    pass


class _ContinueSignal(_ControlFlow):
    pass


class Environment:
    """A scope holding variables, with an optional parent scope.

    Lookups search this scope, then the parent, and so on up to the global scope.

    values: The variables defined directly in this scope.
    parent: The enclosing scope, or None for the global scope.
    """

    values: dict[str, Any]
    parent: "Environment | None"

    def __init__(self, values: dict[str, Any] | None = None, parent: "Environment | None" = None):
        self.values = dict(values or {})
        self.parent = parent

    def child(self, values: dict[str, Any] | None = None) -> "Environment":
        """Creates a new scope inside this one."""
        return Environment(values, self)

    def find(self, name: str) -> "Environment | None":
        """Returns the nearest scope that defines `name`, or None."""
        scope: Environment | None = self
        while scope is not None:
            if name in scope.values:
                return scope
            scope = scope.parent
        return None

    def get(self, name: str) -> Any:
        """Returns the value of a variable. Raises UndefinedVariableError if missing."""
        scope = self.find(name)
        if scope is None:
            raise UndefinedVariableError(name)
        return scope.values[name]

    def define(self, name: str, value: Any) -> None:
        """Creates (or replaces) a variable in this scope."""
        self.values[name] = value

    def assign(self, name: str, value: Any) -> None:
        """Updates the nearest existing variable, or defines it in this scope."""
        scope = self.find(name) or self
        scope.values[name] = value

    def __contains__(self, name: str) -> bool:
        return self.find(name) is not None

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __setitem__(self, name: str, value: Any) -> None:
        self.assign(name, value)

    def __iter__(self) -> Iterator[str]:
        """Iterates over every visible variable name (inner scopes first)."""
        seen: set[str] = set()
        scope: Environment | None = self
        while scope is not None:
            for name in scope.values:
                if name not in seen:
                    seen.add(name)
                    yield name
            scope = scope.parent

    def __repr__(self) -> str:
        return f"Environment({self.values!r}, parent={self.parent!r})"


class Function:
    """A function defined in the language. Calling it runs its body.

    If the body is a Block, the call returns the value of a `return` statement, or
    None. If the body is any other node (like `fn(x) x * 2`), the call returns its value.
    """

    name: str | None
    params: list[str]
    body: Node
    closure: Environment
    interpreter: "Interpreter"

    def __init__(
        self,
        name: str | None,
        params: list[str],
        body: Node,
        closure: Environment,
        interpreter: "Interpreter",
    ):
        self.name = name
        self.params = list(params)
        self.body = body
        self.closure = closure
        self.interpreter = interpreter

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.params):
            raise LanguageRuntimeError(
                f"{self} expects {len(self.params)} argument(s) but got {len(args)}."
            )
        scope = self.closure.child(dict(zip(self.params, args)))
        try:
            result = self.interpreter.execute(self.body, scope)
        except _ReturnSignal as signal:
            return signal.value
        except (_BreakSignal, _ContinueSignal) as signal:
            raise _outside_error(signal) from None
        return None if isinstance(self.body, Block) else result

    def __repr__(self) -> str:
        return f"<function {self.name or '(anonymous)'}>"


def _outside_error(signal: _ControlFlow) -> LanguageRuntimeError:
    where = "a function" if isinstance(signal, _ReturnSignal) else "a loop"
    kind = type(signal.node).__name__.lower()
    return LanguageRuntimeError(f"'{kind}' used outside {where}.", signal.node.line, signal.node.column)


def _power(a: Any, b: Any) -> Any:
    return a**b


class Interpreter(NodeVisitor):
    """Runs syntax trees. See the module docs for how to customize it.

    Class attributes:
        binary_operators: Operator (token type) -> function(left, right).
        unary_operators: Operator (token type) -> function(operand).
        logical_and, logical_or: Operators that short-circuit, like `and` and `or`.
        builtins: Variables every program starts with.

    globals: The global scope.
    env: The scope currently running.
    """

    binary_operators: dict[str, Callable[[Any, Any], Any]] = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "**": _power,
        "^": _power,
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }
    unary_operators: dict[str, Callable[[Any], Any]] = {
        "-": operator.neg,
        "+": operator.pos,
        "!": operator.not_,
        "not": operator.not_,
    }
    logical_and: frozenset[str] = frozenset({"and", "&&"})
    logical_or: frozenset[str] = frozenset({"or", "||"})
    builtins: dict[str, Any] = {
        "print": print,
        "input": input,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "abs": abs,
        "min": min,
        "max": max,
    }

    globals: Environment
    env: Environment

    def __init__(self, builtins: dict[str, Any] | None = None):
        """builtins: Extra variables to start with (added to the class's `builtins`)."""
        self.globals = Environment({**self.builtins, **(builtins or {})})
        self.env = self.globals

    # --- Running ---

    def run(self, node: Node) -> Any:
        """Runs a tree in the global scope and returns its value."""
        self.env = self.globals
        try:
            return self.visit(node)
        except _ControlFlow as signal:
            raise _outside_error(signal) from None

    def visit(self, node: Node) -> Any:
        """Runs one node. Python errors are turned into LanguageRuntimeError with the
        node's position, so builtins can simply raise normal exceptions."""
        try:
            return super().visit(node)
        except LanguageError as error:
            if error.line is None and node.token is not None:
                error.line, error.column = node.line, node.column
            raise
        except NotImplementedError:
            raise
        except Exception as error:
            message = str(error) or type(error).__name__
            raise LanguageRuntimeError(message, node.line, node.column) from error

    def execute(self, node: Node, env: Environment) -> Any:
        """Runs a node in the given scope, then switches back to the current one."""
        previous = self.env
        self.env = env
        try:
            return self.visit(node)
        finally:
            self.env = previous

    def execute_statements(self, statements: list[Node], env: Environment) -> Any:
        """Runs statements in the given scope and returns the last one's value."""
        previous = self.env
        self.env = env
        try:
            result = None
            for statement in statements:
                result = self.visit(statement)
            return result
        finally:
            self.env = previous

    def is_truthy(self, value: Any) -> bool:
        """Decides whether a value counts as true in conditions. Uses Python's rules."""
        return bool(value)

    def call(self, function: Any, args: list[Any]) -> Any:
        """Calls a function value with already-evaluated arguments."""
        if not callable(function):
            raise LanguageRuntimeError(f"{function!r} is not callable.")
        return function(*args)

    # --- Ready-made nodes ---

    def visit_Program(self, node: Program) -> Any:
        return self.execute_statements(node.statements, self.env)

    def visit_Block(self, node: Block) -> Any:
        return self.execute_statements(node.statements, self.env.child())

    def visit_Literal(self, node: Literal) -> Any:
        return node.value

    def visit_Name(self, node: Name) -> Any:
        return self.env.get(node.name)

    def visit_Assign(self, node: Assign) -> Any:
        value = self.visit(node.value)
        if node.declare:
            self.env.define(node.name, value)
        else:
            self.env.assign(node.name, value)
        return value

    def visit_UnaryOp(self, node: UnaryOp) -> Any:
        function = self.unary_operators.get(node.op)
        if function is None:
            raise LanguageRuntimeError(f"Unknown unary operator {node.op!r}.")
        return function(self.visit(node.operand))

    def visit_BinaryOp(self, node: BinaryOp) -> Any:
        if node.op in self.logical_and:
            left = self.visit(node.left)
            return self.visit(node.right) if self.is_truthy(left) else left
        if node.op in self.logical_or:
            left = self.visit(node.left)
            return left if self.is_truthy(left) else self.visit(node.right)
        function = self.binary_operators.get(node.op)
        if function is None:
            raise LanguageRuntimeError(f"Unknown binary operator {node.op!r}.")
        return function(self.visit(node.left), self.visit(node.right))

    def visit_Call(self, node: Call) -> Any:
        function = self.visit(node.callee)
        args = [self.visit(arg) for arg in node.args]
        return self.call(function, args)

    def visit_If(self, node: If) -> Any:
        if self.is_truthy(self.visit(node.condition)):
            return self.visit(node.then)
        if node.otherwise is not None:
            return self.visit(node.otherwise)
        return None

    def visit_While(self, node: While) -> None:
        while self.is_truthy(self.visit(node.condition)):
            try:
                self.visit(node.body)
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue

    def visit_Break(self, node: Break) -> None:
        raise _BreakSignal(node)

    def visit_Continue(self, node: Continue) -> None:
        raise _ContinueSignal(node)

    def visit_FunctionDef(self, node: FunctionDef) -> Function:
        function = Function(node.name, node.params, node.body, self.env, self)
        if node.name is not None:
            self.env.define(node.name, function)
        return function

    def visit_Return(self, node: Return) -> None:
        value = None if node.value is None else self.visit(node.value)
        raise _ReturnSignal(node, value)
