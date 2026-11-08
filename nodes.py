"""Syntax tree nodes

A parser turns tokens into a tree of `Node` objects, and an interpreter walks that
tree to run the program. This module has the `Node` base class, a set of common
ready-made nodes, and `NodeVisitor` for walking trees.

Make your own nodes by subclassing `Node` as a dataclass:

    @dataclasses.dataclass
    class Print(Node):
        value: Node

Every node also takes an optional keyword-only `token`, the token it came from, which
is used to report where errors happen:

    Print(Literal(1), token=print_token)
"""

import dataclasses
from collections.abc import Iterator
from typing import Any

from .tokenizer import Token


@dataclasses.dataclass
class Node:
    """Base class for syntax tree nodes."""

    token: Token | None = dataclasses.field(
        default=None, kw_only=True, compare=False, repr=False
    )

    @property
    def line(self) -> int | None:
        """The line this node starts on, if known."""
        return self.token.line if self.token else None

    @property
    def column(self) -> int | None:
        """The column this node starts on, if known."""
        return self.token.column if self.token else None

    def children(self) -> Iterator["Node"]:
        """Yields the nodes directly inside this one, in field order.

        Looks inside lists and tuples too, so `Block.statements` works.
        """
        for field in dataclasses.fields(self):
            if field.name == "token":
                continue
            value = getattr(self, field.name)
            if isinstance(value, Node):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (item for item in value if isinstance(item, Node))

    def walk(self) -> Iterator["Node"]:
        """Yields this node and every node below it (depth first, parents first)."""
        stack: list[Node] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(list(node.children())))

    def dump(self, indent: str = "  ") -> str:
        """Returns the tree as an indented, human-readable string."""
        return "\n".join(self._dump_lines(indent, 0))

    def _dump_lines(self, indent: str, depth: int) -> list[str]:
        pad = indent * depth
        simple: list[str] = []
        nested: list[tuple[str, Any]] = []
        for field in dataclasses.fields(self):
            if field.name == "token":
                continue
            value = getattr(self, field.name)
            is_nodes = isinstance(value, (list, tuple)) and any(
                isinstance(item, Node) for item in value
            )
            if isinstance(value, Node) or is_nodes:
                nested.append((field.name, value))
            else:
                simple.append(f"{field.name}={value!r}")
        lines = [f"{pad}{type(self).__name__}({', '.join(simple)})"]
        for name, value in nested:
            lines.append(f"{pad}{indent}{name}:")
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                if isinstance(item, Node):
                    lines.extend(item._dump_lines(indent, depth + 2))
                else:
                    lines.append(f"{pad}{indent * 2}{item!r}")
        return lines


# --- Ready-made nodes -------------------------------------------------------
# The default `Parser` and `Interpreter` know how to build and run these.


@dataclasses.dataclass
class Program(Node):
    """The whole program: a list of statements run in the global scope."""

    statements: list[Node]


@dataclasses.dataclass
class Block(Node):
    """A list of statements run in a new inner scope."""

    statements: list[Node]


@dataclasses.dataclass
class Literal(Node):
    """A fixed value, like a number or string."""

    value: Any


@dataclasses.dataclass
class Name(Node):
    """A variable lookup."""

    name: str


@dataclasses.dataclass
class UnaryOp(Node):
    """A prefix operator, like `-x` or `not x`."""

    op: str
    operand: Node


@dataclasses.dataclass
class BinaryOp(Node):
    """An infix operator, like `a + b`."""

    op: str
    left: Node
    right: Node


@dataclasses.dataclass
class Call(Node):
    """A function call, like `f(a, b)`."""

    callee: Node
    args: list[Node]


@dataclasses.dataclass
class Assign(Node):
    """Sets a variable.

    declare: If True, always creates the variable in the current scope (like `let x = 1`).
             If False, updates the nearest existing variable, or creates it in the
             current scope if there is none.
    """

    name: str
    value: Node
    declare: bool = False


@dataclasses.dataclass
class If(Node):
    """Runs `then` if `condition` is truthy, otherwise `otherwise` (if given)."""

    condition: Node
    then: Node
    otherwise: Node | None = None


@dataclasses.dataclass
class While(Node):
    """Runs `body` as long as `condition` is truthy."""

    condition: Node
    body: Node


@dataclasses.dataclass
class Break(Node):
    """Leaves the innermost loop."""


@dataclasses.dataclass
class Continue(Node):
    """Skips to the next iteration of the innermost loop."""


@dataclasses.dataclass
class FunctionDef(Node):
    """Defines a function.

    name: The variable the function is stored in, or None for an anonymous function
          (the node then evaluates to the function).
    """

    name: str | None
    params: list[str]
    body: Node


@dataclasses.dataclass
class Return(Node):
    """Returns from the current function, with `value` or None."""

    value: Node | None = None


# --- Visiting ---------------------------------------------------------------


class NodeVisitor:
    """Walks a syntax tree by calling a `visit_<ClassName>` method for each node.

    If a node's class has no method, its parent classes are tried (so a subclass of
    `BinaryOp` uses `visit_BinaryOp`), and finally `generic_visit`.
    """

    def visit(self, node: Node) -> Any:
        """Visits a node and returns the result of its visit method."""
        return self._visitor_for(type(node))(node)

    def generic_visit(self, node: Node) -> Any:
        """Called for nodes without a visit method. Raises NotImplementedError by default."""
        raise NotImplementedError(
            f"{type(self).__name__} has no visit_{type(node).__name__} method."
        )

    def _visitor_for(self, node_class: type) -> Any:
        cache = self.__dict__.setdefault("_visitor_cache", {})
        method = cache.get(node_class)
        if method is None:
            method = self.generic_visit
            for cls in node_class.__mro__:
                found = getattr(self, f"visit_{cls.__name__}", None)
                if found is not None:
                    method = found
                    break
            cache[node_class] = method
        return method
