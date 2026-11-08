# langpy

A small library for making your own programming languages in Python.

A language is made of three classes, each of which you can subclass:

| Part | Job |
| --- | --- |
| `Tokenizer` | Splits source code into tokens, using `ValidToken` rules. |
| `Parser` | Turns tokens into a tree of `Node`s. |
| `Interpreter` | Runs the tree. |

`Language` ties them together and gives you `run`, `run_file`, a REPL and a command line.

## Install

```
pip install -e .            # from this folder
pip install -e .[test]      # with pytest
```

## A calculator in 30 lines

```python
from langpy import Assign, Language, Parser, parse_number
from langpy import ValidToken as T


class CalcParser(Parser):
    literals = {"NUMBER": parse_number}
    identifier = "IDENT"
    binary_operators = {"+": 10, "-": 10, "*": 20, "/": 20, "^": 30}
    right_associative = {"^"}
    unary_operators = {"-": 25}
    grouping = ("(", ")")
    call = ("(", ",", ")")
    statement_separator = ";"

    def parse_statement(self):
        if self.match("let"):
            name = self.expect("IDENT")
            self.expect("=")
            value = self.parse_expression()
            self.end_statement()
            return Assign(name.value, value, declare=True, token=name)
        return super().parse_statement()


class Calc(Language):
    name = "calc"
    tokens = [
        *(T.exact(s) for s in ["let", "=", "+", "-", "*", "/", "^", "(", ")", ",", ";"]),
        T.regex(r"[A-Za-z_]\w*", name="IDENT"),
        T.regex(r"[0-9]+(\.[0-9]+)?", name="NUMBER"),
        T.regex(r"#[^\n]*", skip=True),  # comments
    ]
    parser_class = CalcParser


calc = Calc()
calc.run("let x = 2 ^ 3;")
print(calc.run("max(x, 5) * -1"))   # -8  (max is a built-in)

if __name__ == "__main__":
    raise SystemExit(calc.main())     # `python calc.py file.calc`, or a REPL with no file
```

See [examples/mini.py](examples/mini.py) for a full scripting language with `if`, `while`,
functions, closures, `break`/`continue` and strings. Try it with
`python -m langpy.examples.mini`.

## Tokenizer

- `ValidToken.exact("let")` matches exact text. Its token type is the text itself.
- `ValidToken.regex(r"[0-9]+", name="NUMBER")` matches a regex.
- `skip=True` drops matches (comments).
- The longest match wins. On a tie, exact rules beat regex rules, so keywords beat identifiers.
- Whitespace is skipped unless `skip_whitespace=False`.

## Parser

Subclass `Parser`. Expressions use these class attributes (all optional):

| Attribute | Example | Builds |
| --- | --- | --- |
| `literals` | `{"NUMBER": parse_number, "true": True}` | `Literal` |
| `identifier` | `"IDENT"` | `Name` |
| `binary_operators` | `{"+": 10, "*": 20}` (higher binds tighter) | `BinaryOp` |
| `right_associative` | `{"^"}` | |
| `unary_operators` | `{"-": 25}` | `UnaryOp` |
| `grouping` | `("(", ")")` | |
| `call` | `("(", ",", ")")` | `Call` |
| `statement_separator` | `";"` | |

For custom expression syntax, add decorated methods:

```python
from langpy import infix, prefix

class MyParser(Parser):
    @prefix("[")                       # expressions that start with "["
    def list_literal(self, token):
        return ListNode(self.parse_delimited("]"), token=token)

    @infix("[", precedence=100)        # "[" after an expression
    def index(self, left, token):
        key = self.parse_expression()
        self.expect("]")
        return Index(left, key, token=token)
```

For statements, override `parse_statement`. Helpers: `peek`, `advance`, `check`, `match`,
`expect`, `at_end`, `error`, `parse_expression`, `parse_block`, `parse_delimited`,
`parse_statements`, `end_statement`.

## Nodes

Ready-made nodes: `Program`, `Block`, `Literal`, `Name`, `UnaryOp`, `BinaryOp`, `Call`,
`Assign`, `If`, `While`, `Break`, `Continue`, `FunctionDef`, `Return`. The default
interpreter runs all of them.

Make your own as dataclasses. Pass `token=` so errors can point at the right place:

```python
import dataclasses
from langpy import Node

@dataclasses.dataclass
class Index(Node):
    target: Node
    key: Node
```

Every node has `children()`, `walk()` and `dump()` (a readable tree).

## Interpreter

Subclass `Interpreter` and add `visit_<NodeClass>` methods for your nodes. Change the
operator tables or built-in variables with class attributes:

```python
import math
from langpy import Interpreter

class MyInterpreter(Interpreter):
    binary_operators = {**Interpreter.binary_operators, "..": lambda a, b: f"{a}{b}"}
    builtins = {**Interpreter.builtins, "sqrt": math.sqrt}

    def visit_Index(self, node):
        return self.visit(node.target)[self.visit(node.key)]
```

Any Python callable works as a function in your language, and functions defined in
your language can be called from Python:

```python
lang.interpreter.globals.define("now", time.time)
lang.run("fn add(a, b) { return a + b; }")
lang.interpreter.globals["add"](1, 2)   # 3
```

Variables live in `Environment` scopes. `Language.run` keeps the same interpreter
between calls; `Language.reset()` starts fresh.

## Errors

Everything raised for bad input is a `LanguageError` with `line` and `column`:

- `TokenizerError` → `InvalidTokenError`
- `ParseError` → `UnexpectedTokenError`, `UnexpectedEOFError`
- `LanguageRuntimeError` → `UndefinedVariableError`

Python errors during a run (like `ZeroDivisionError`) become `LanguageRuntimeError`
at the node that caused them. `error.format(source)` gives a readable report:

```
Line 2, column 13: Variable 'z' is not defined.
    let y = x + z;
                ^
```

## Tests

```
.venv\Scripts\python.exe -m pytest
```
