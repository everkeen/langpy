"""A complete language: tokenizer rules + parser + interpreter

Example:
    class Calc(Language):
        name = "calc"
        tokens = [ValidToken.exact("+"), ValidToken.regex(r"[0-9]+", name="NUMBER")]
        parser_class = CalcParser
        interpreter_class = Interpreter

    calc = Calc()
    calc.run("1 + 2")    # 3
    calc.repl()          # interactive prompt

    if __name__ == "__main__":
        raise SystemExit(Calc().main())   # `python calc.py file.calc` or a REPL

You can also skip subclassing:
    calc = Language(tokens=RULES, parser_class=CalcParser)
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .errors import LanguageError
from .interpreter import Interpreter
from .nodes import Assign, FunctionDef, Node, Program
from .parser import Parser, UnexpectedEOFError
from .tokenizer import Token, Tokenizer, ValidToken


class Language:
    """Ties together token rules, a parser class and an interpreter class.

    The interpreter is kept between `run` calls, so variables defined by one call
    are still there in the next one (like in a REPL). Call `reset` to start over.

    Class attributes (or the matching constructor arguments):
        name: Shown in the REPL banner.
        tokens: The tokenizer rules.
        skip_whitespace: Passed to the Tokenizer.
        parser_class: A `Parser` subclass.
        interpreter_class: An `Interpreter` subclass.
        prompt, continuation_prompt: REPL prompts.
    """

    name: str = "langpy"
    tokens: list[ValidToken] = []
    skip_whitespace: bool = True
    parser_class: type[Parser] = Parser
    interpreter_class: type[Interpreter] = Interpreter
    prompt: str = ">>> "
    continuation_prompt: str = "... "

    interpreter: Interpreter

    def __init__(
        self,
        tokens: list[ValidToken] | None = None,
        parser_class: type[Parser] | None = None,
        interpreter_class: type[Interpreter] | None = None,
        name: str | None = None,
    ):
        if tokens is not None:
            self.tokens = list(tokens)
        if parser_class is not None:
            self.parser_class = parser_class
        if interpreter_class is not None:
            self.interpreter_class = interpreter_class
        if name is not None:
            self.name = name
        self.interpreter = self.create_interpreter()

    # --- Factories (override to customize how each part is made) ---

    def create_tokenizer(self, source: str) -> Tokenizer:
        return Tokenizer(source, self.tokens, skip_whitespace=self.skip_whitespace)

    def create_parser(self, source: str) -> Parser:
        return self.parser_class(self.create_tokenizer(source))

    def create_interpreter(self) -> Interpreter:
        return self.interpreter_class()

    # --- Pipeline ---

    def tokenize(self, source: str) -> list[Token]:
        """Returns all tokens in `source`."""
        return self.create_tokenizer(source).run()

    def parse(self, source: str) -> Node:
        """Returns the syntax tree for `source`."""
        return self.create_parser(source).parse()

    def run(self, source: str) -> Any:
        """Parses and runs `source`, returning the value of the last statement.

        Raises a LanguageError (TokenizerError, ParseError or LanguageRuntimeError)
        if anything goes wrong. Use `error.format(source)` for a readable message.
        """
        return self.interpreter.run(self.parse(source))

    def run_file(self, path: str | Path, encoding: str = "utf-8") -> Any:
        """Runs a source file."""
        return self.run(Path(path).read_text(encoding=encoding))

    def reset(self) -> None:
        """Forgets all variables by creating a fresh interpreter."""
        self.interpreter = self.create_interpreter()

    def format_result(self, tree: Node, value: Any) -> str | None:
        """How the REPL shows the result of running `tree`. Return None to show nothing.

        By default hides None, and the results of assignments and named functions.
        """
        last = tree.statements[-1] if isinstance(tree, Program) and tree.statements else tree
        if value is None or isinstance(last, Assign):
            return None
        if isinstance(last, FunctionDef) and last.name is not None:
            return None
        return repr(value)

    # --- Command line ---

    def repl(
        self,
        read: Callable[[str], str] = input,
        write: Callable[[str], Any] = print,
    ) -> None:
        """Runs an interactive prompt until end of input (Ctrl+D / Ctrl+Z) or `exit`.

        Input that ends too early (like an unclosed `{`) continues on the next line.
        `read` and `write` can be replaced, e.g. for testing or a GUI.
        """
        write(f"{self.name} REPL. Type 'exit' or press Ctrl+D (Ctrl+Z on Windows) to quit.")
        while True:
            try:
                source = read(self.prompt)
            except (EOFError, KeyboardInterrupt):
                write("")
                return
            if source.strip() == "exit":
                return
            if not source.strip():
                continue

            while True:
                try:
                    tree = self.parse(source)
                    value = self.interpreter.run(tree)
                except UnexpectedEOFError:
                    try:
                        source += "\n" + read(self.continuation_prompt)
                    except (EOFError, KeyboardInterrupt):
                        write("")
                        break
                    continue
                except LanguageError as error:
                    write(error.format(source))
                else:
                    shown = self.format_result(tree, value)
                    if shown is not None:
                        write(shown)
                break

    def main(self, argv: list[str] | None = None, stderr: TextIO | None = None) -> int:
        """Command line entry point: runs the file named in `argv`, or a REPL if none.

        Returns an exit code (0 on success, 1 on a language error).
        """
        argv = sys.argv[1:] if argv is None else argv
        stderr = stderr or sys.stderr
        if not argv:
            self.repl()
            return 0
        path = Path(argv[0])
        source = path.read_text(encoding="utf-8")
        try:
            self.run(source)
        except LanguageError as error:
            print(f"{path}: {error.format(source)}", file=stderr)
            return 1
        return 0
