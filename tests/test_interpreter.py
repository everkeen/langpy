"""Tests for the interpreter, the Language class and the Mini example language."""

import dataclasses
import io
import math

import pytest

from langpy import (
    Environment,
    Interpreter,
    Language,
    LanguageError,
    LanguageRuntimeError,
    Literal,
    Node,
    NodeVisitor,
    Parser,
    Program,
    UndefinedVariableError,
    UnexpectedEOFError,
    prefix,
)
from langpy import ValidToken as T
from langpy.examples.mini import Mini, MiniParser


def run(source: str):
    return Mini().run(source)


# --- Environment --------------------------------------------------------------


def test_environment_scopes():
    outer = Environment({"x": 1})
    inner = outer.child({"y": 2})
    assert inner.get("x") == 1 and inner["y"] == 2
    assert "x" in inner and "y" not in outer
    inner.assign("x", 10)  # updates the outer variable
    assert outer["x"] == 10
    inner.define("x", 20)  # shadows it
    assert inner["x"] == 20 and outer["x"] == 10
    inner["z"] = 3  # new variables go in the current scope
    assert "z" not in outer
    assert sorted(inner) == ["x", "y", "z"]
    with pytest.raises(UndefinedVariableError):
        outer.get("nope")


# --- Running Mini programs ----------------------------------------------------


def test_arithmetic_and_precedence():
    assert run("1 + 2 * 3") == 7
    assert run("(1 + 2) * 3") == 9
    assert run("10 - 3 - 2") == 5
    assert run("-2 * 3") == -6
    assert run("7 % 3") == 1
    assert run("1 / 2") == 0.5


def test_strings_and_booleans():
    assert run('"ab" + "cd"') == "abcd"
    assert run('"a\\"b"') == 'a"b'
    assert run("1 < 2 and 2 < 3") is True
    assert run("not true or false") is False
    assert run("nil") is None


def test_logical_operators_short_circuit():
    # `crash` is undefined, so evaluating it would raise.
    assert run("false and crash") is False
    assert run("true or crash") is True
    assert run("0 or 5") == 5


def test_variables_and_scopes():
    assert run("let x = 1; x = x + 1; x") == 2
    assert run("let x = 1; if true { let x = 2; } x") == 1  # inner `let` shadows
    assert run("let x = 1; if true { x = 2; } x") == 2  # plain `=` updates


def test_if_else_chain():
    source = """
    fn sign(n) {
        if n < 0 { return -1; } else if n == 0 { return 0; } else { return 1; }
    }
    sign(-5) + sign(0) * 10 + sign(7) * 100
    """
    assert run(source) == 99


def test_while_break_continue():
    source = """
    let total = 0;
    let i = 0;
    while true {
        i = i + 1;
        if i > 10 { break; }
        if i % 2 == 0 { continue; }
        total = total + i;
    }
    total
    """
    assert run(source) == 1 + 3 + 5 + 7 + 9


def test_recursion():
    source = "fn fib(n) { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); } fib(15)"
    assert run(source) == 610


def test_closures():
    source = """
    fn counter() {
        let count = 0;
        return fn() { count = count + 1; return count; };
    }
    let c = counter();
    c(); c();
    c()
    """
    assert run(source) == 3


def test_expression_bodied_function():
    assert run("let double = fn(x) x * 2; double(21)") == 42
    assert run("fn add(a, b) a + b; add(1, 2)") == 3


def test_block_function_without_return_gives_nil():
    assert run("fn f() { 1; } f()") is None


def test_builtins(capsys):
    run('print("hi", 1 + 1)')
    assert capsys.readouterr().out == "hi 2\n"
    assert run('len("abc")') == 3


def test_python_functions_can_be_added():
    mini = Mini()
    mini.interpreter.globals.define("sqrt", math.sqrt)
    assert mini.run("sqrt(16)") == 4


def test_language_functions_can_be_called_from_python():
    mini = Mini()
    mini.run("fn add(a, b) { return a + b; }")
    assert mini.interpreter.globals["add"](2, 3) == 5


def test_state_persists_until_reset():
    mini = Mini()
    mini.run("let x = 5;")
    assert mini.run("x * 2") == 10
    mini.reset()
    with pytest.raises(UndefinedVariableError):
        mini.run("x")


# --- Runtime errors -----------------------------------------------------------


def test_undefined_variable_has_position():
    with pytest.raises(UndefinedVariableError) as err:
        run("let a = 1;\nb + 1")
    assert (err.value.line, err.value.column) == (2, 1)
    assert err.value.name == "b"


def test_python_errors_become_language_errors():
    with pytest.raises(LanguageRuntimeError) as err:
        run("let x = 1;\nx / 0")
    assert (err.value.line, err.value.column) == (2, 3)
    assert isinstance(err.value.__cause__, ZeroDivisionError)


def test_type_error_from_operator():
    with pytest.raises(LanguageRuntimeError):
        run('1 + "a"')


def test_wrong_argument_count():
    with pytest.raises(LanguageRuntimeError, match="expects 2 argument"):
        run("fn f(a, b) { } f(1)")


def test_calling_a_non_function():
    with pytest.raises(LanguageRuntimeError, match="not callable"):
        run("let x = 1; x()")


@pytest.mark.parametrize("source", ["return 1;", "break;", "continue;", "fn f() { break; } f()"])
def test_control_flow_outside_its_place(source):
    with pytest.raises(LanguageRuntimeError, match="outside"):
        run(source)


def test_deep_recursion_is_a_language_error():
    with pytest.raises(LanguageRuntimeError):
        run("fn f(n) { return f(n + 1); } f(0)")


def test_error_format_shows_source_line():
    source = "let x = 1;\nlet y = x + z;"
    with pytest.raises(LanguageError) as err:
        run(source)
    assert err.value.format(source) == (
        "Line 2, column 13: Variable 'z' is not defined.\n"
        "    let y = x + z;\n"
        "                ^"
    )


def test_error_format_without_position():
    assert LanguageError("oops").format() == "LanguageError: oops"


# --- Customizing the interpreter ----------------------------------------------


@dataclasses.dataclass
class Shout(Node):
    value: Node


class ShoutParser(MiniParser):
    @prefix("IDENT")
    def maybe_shout(self, token):
        if token.value == "shout":
            return Shout(self.parse_expression(100), token=token)
        return self._name_rule(token)


class ShoutInterpreter(Interpreter):
    binary_operators = {**Interpreter.binary_operators, "+": lambda a, b: f"{a}{b}"}
    builtins = {"answer": 42}

    def visit_Shout(self, node):
        return str(self.visit(node.value)).upper() + "!"


def test_custom_nodes_and_operators():
    lang = Mini(parser_class=ShoutParser, interpreter_class=ShoutInterpreter)
    assert lang.run('shout "hey"') == "HEY!"
    assert lang.run("1 + 2") == "12"
    assert lang.run("answer") == 42
    with pytest.raises(UndefinedVariableError):
        lang.run("print")  # builtins were replaced


def test_extra_builtins_in_constructor():
    interpreter = Interpreter(builtins={"seven": 7})
    assert interpreter.globals["seven"] == 7
    assert interpreter.globals["print"] is print


def test_unknown_node_type():
    @dataclasses.dataclass
    class Mystery(Node):
        pass

    with pytest.raises(NotImplementedError):
        Interpreter().run(Program([Mystery()]))


def test_visitor_falls_back_to_parent_class():
    class Special(Literal):
        pass

    assert Interpreter().run(Special(5)) == 5


def test_node_visitor_and_walk():
    tree = Mini().parse("let x = f(1 + 2);")

    class Counter(NodeVisitor):
        def __init__(self):
            self.literals = 0

        def visit_Node(self, node):
            for child in node.children():
                self.visit(child)

        def visit_Literal(self, node):
            self.literals += 1

    counter = Counter()
    counter.visit(tree)
    assert counter.literals == 2
    names = [type(node).__name__ for node in tree.walk()]
    assert names == ["Program", "Assign", "Call", "Name", "BinaryOp", "Literal", "Literal"]


# --- Language -----------------------------------------------------------------


def test_language_without_subclassing():
    class Adder(Parser):
        literals = {"N": int}
        binary_operators = {"+": 1}

    lang = Language(tokens=[T.exact("+"), T.regex(r"\d+", name="N")], parser_class=Adder)
    assert lang.run("1 + 2 + 3") == 6
    assert [t.value for t in lang.tokenize("1+2")] == ["1", "+", "2"]


def test_run_file(tmp_path):
    path = tmp_path / "prog.mini"
    path.write_text("let x = 6; x * 7", encoding="utf-8")
    assert Mini().run_file(path) == 42


def test_main_runs_file(tmp_path, capsys):
    path = tmp_path / "prog.mini"
    path.write_text('print("from file");', encoding="utf-8")
    assert Mini().main([str(path)]) == 0
    assert capsys.readouterr().out == "from file\n"


def test_main_reports_errors(tmp_path):
    path = tmp_path / "bad.mini"
    path.write_text("let x = ;", encoding="utf-8")
    stderr = io.StringIO()
    assert Mini().main([str(path)], stderr=stderr) == 1
    assert "Expected an expression" in stderr.getvalue()


def test_unexpected_eof_lets_repl_continue():
    with pytest.raises(UnexpectedEOFError):
        Mini().parse("fn f() {")


def test_repl():
    lines = iter(["let x = 2;", "x * 21", "fn f() {", "  return 1;", "}", "f()", "y", "", "exit"])
    output: list[str] = []
    Mini().repl(read=lambda prompt: next(lines), write=output.append)
    assert output[1:] == [
        "42",
        "1",
        "Line 1, column 1: Variable 'y' is not defined.\n    y\n    ^",
    ]


def test_repl_stops_at_end_of_input():
    def read(prompt):
        raise EOFError

    output: list[str] = []
    Mini().repl(read=read, write=output.append)
    assert output[-1] == ""
