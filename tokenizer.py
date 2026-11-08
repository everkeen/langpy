"""A tokenizer for the simple programming language

Breaks the source code into individual tokens for further processing.

Example:
    rules = [
        ValidToken.exact("let"),
        ValidToken.exact("="),
        ValidToken.regex(r"[A-Za-z_]\\w*", name="IDENT"),
        ValidToken.regex(r"[0-9]+", name="NUMBER"),
        ValidToken.regex(r"#[^\\n]*", skip=True),  # comments
    ]
    for token in Tokenizer("let x = 1  # one", rules):
        print(token.type, token.value, token.line, token.column)
"""

import dataclasses
import enum
import re
from collections.abc import Iterator

from .errors import LanguageError


class TokenizerError(LanguageError):
    """An error related to tokenization."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(message, line, column)


class InvalidTokenError(TokenizerError):
    """An error related to an invalid token."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(message, line, column)


class ReachedEOFError(TokenizerError):
    """An error related to reaching the end of the source code."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(message, line, column)


class ValidTokenRule(enum.Enum):
    """How a valid token is matched."""

    EXACT = "exact"
    REGEX = "regex"


@dataclasses.dataclass(frozen=True)
class ValidToken:
    """A rule describing one kind of token.

    rule: How `pattern` is matched: exact text, or a regular expression.
    pattern: The text or regex to match.
    name: The token type given to matches. Defaults to the pattern itself.
    skip: If True, matches are consumed but never returned (useful for comments).
    flags: `re` flags for regex rules, e.g. `re.IGNORECASE`.

    Regexes are matched at the tokenizer's current position, so they never skip ahead.
    A regex match of zero characters counts as no match.
    """

    rule: ValidTokenRule
    pattern: str
    name: str = None  # type: ignore[assignment]  # filled in by __post_init__
    skip: bool = False
    flags: int = 0
    _compiled: "re.Pattern[str] | None" = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if not self.pattern:
            raise ValueError("A token pattern cannot be empty.")
        if self.name is None:
            object.__setattr__(self, "name", self.pattern)
        if self.rule == ValidTokenRule.REGEX:
            object.__setattr__(self, "_compiled", re.compile(self.pattern, self.flags))

    @classmethod
    def exact(cls, pattern: str, name: str | None = None, skip: bool = False) -> "ValidToken":
        """Creates a rule that matches `pattern` exactly."""
        return cls(ValidTokenRule.EXACT, pattern, name, skip)  # type: ignore[arg-type]

    @classmethod
    def regex(
        cls, pattern: str, name: str | None = None, skip: bool = False, flags: int = 0
    ) -> "ValidToken":
        """Creates a rule that matches the regular expression `pattern`."""
        return cls(ValidTokenRule.REGEX, pattern, name, skip, flags)  # type: ignore[arg-type]

    def match(self, source: str, position: int) -> str | None:
        """Returns the text this rule matches at `position`, or None if it doesn't match."""
        if self.rule == ValidTokenRule.EXACT:
            return self.pattern if source.startswith(self.pattern, position) else None
        assert self._compiled is not None
        found = self._compiled.match(source, position)
        if found is None or found.end() == position:
            return None
        return found.group(0)


@dataclasses.dataclass(frozen=True)
class Token:
    """A token found in the source code.

    type: The `name` of the rule that matched it.
    value: The matched text.
    line, column: Where the token starts (both start at 1).
    position: Where the token starts as an index into the source code.
    """

    type: str
    value: str
    line: int
    column: int
    position: int


class Tokenizer:
    """Tokenizes source code into individual tokens based on a list of valid tokens.

    At each position the longest match wins. If several rules match the same length,
    exact rules beat regex rules (so keywords beat an identifier regex), and after that
    the rule listed first wins.

    Whitespace between tokens is skipped unless `skip_whitespace` is False, in which
    case you need rules for it (e.g. a NEWLINE token for indentation-based languages).
    """

    source_code: str
    valid_tokens: list[ValidToken]
    skip_whitespace: bool
    position: int
    current_token: Token | None
    _line: int
    _column: int

    def __init__(
        self, source_code: str, valid_tokens: list[ValidToken], skip_whitespace: bool = True
    ):
        self.source_code = source_code
        self.valid_tokens = list(valid_tokens)
        self.skip_whitespace = skip_whitespace
        self.reset()

    @property
    def line(self) -> int:
        """The current line (starts at 1)."""
        return self._line

    @property
    def column(self) -> int:
        """The current column (starts at 1)."""
        return self._column

    def _advance(self, length: int) -> str:
        """Advances the tokenizer position by the given length and returns the skipped text."""
        end = self.position + length
        if end > len(self.source_code):
            raise ReachedEOFError(
                "Advancing beyond the end of the source code.", self._line, self._column
            )
        text = self.source_code[self.position : end]
        newlines = text.count("\n")
        if newlines:
            self._line += newlines
            self._column = length - text.rfind("\n")
        else:
            self._column += length
        self.position = end
        return text

    def _peek_char(self, offset: int = 0) -> str | None:
        """Returns the character at an offset position without advancing the tokenizer."""
        position = self.position + offset
        if position >= len(self.source_code):
            return None
        return self.source_code[position]

    def _get_char(self) -> str | None:
        """Returns the character at the current position."""
        return self._peek_char(0)

    def _skip_whitespace(self) -> None:
        """Advances past any whitespace at the current position."""
        end = self.position
        while end < len(self.source_code) and self.source_code[end].isspace():
            end += 1
        if end > self.position:
            self._advance(end - self.position)

    def _find_match(self) -> tuple[ValidToken, str] | None:
        """Finds the best rule matching at the current position."""
        best: tuple[ValidToken, str] | None = None
        for rule in self.valid_tokens:
            text = rule.match(self.source_code, self.position)
            if text is None:
                continue
            if best is None:
                best = (rule, text)
                continue
            best_rule, best_text = best
            longer = len(text) > len(best_text)
            exact_beats_regex = (
                len(text) == len(best_text)
                and rule.rule == ValidTokenRule.EXACT
                and best_rule.rule == ValidTokenRule.REGEX
            )
            if longer or exact_beats_regex:
                best = (rule, text)
        return best

    def next_token(self) -> Token | None:
        """Returns the next token from the source code, or None at the end."""
        while True:
            if self.skip_whitespace:
                self._skip_whitespace()
            if self.position >= len(self.source_code):
                return None

            found = self._find_match()
            if found is None:
                raise InvalidTokenError(
                    f"Unexpected character {self._get_char()!r}.", self._line, self._column
                )

            rule, text = found
            token = Token(rule.name, text, self._line, self._column, self.position)
            self._advance(len(text))
            if not rule.skip:
                self.current_token = token
                return token

    def peek_token(self, offset: int = 0) -> Token | None:
        """Returns an upcoming token without advancing the tokenizer.

        `offset` 0 is the next token, 1 the one after it, and so on.
        """
        saved = (self.position, self._line, self._column, self.current_token)
        try:
            token = None
            for _ in range(offset + 1):
                token = self.next_token()
                if token is None:
                    break
            return token
        finally:
            self.position, self._line, self._column, self.current_token = saved

    def __iter__(self) -> Iterator[Token]:
        return self

    def __next__(self) -> Token:
        token = self.next_token()
        if token is None:
            raise StopIteration
        return token

    def reset(self) -> None:
        """Resets the tokenizer to the beginning of the source code."""
        self.position = 0
        self.current_token = None
        self._line = 1
        self._column = 1

    def set_source_code(self, source_code: str) -> None:
        """Sets a new source code for the tokenizer."""
        self.source_code = source_code
        self.reset()

    def get_current_token(self) -> Token | None:
        """Returns the most recently returned token."""
        return self.current_token

    def generate_tokens(self) -> Iterator[Token]:
        """Generator for tokens (Same as iterating directly)."""
        while True:
            token = self.next_token()
            if token is None:
                break
            yield token

    def run(self) -> list[Token]:
        """Runs the tokenizer and returns a list of all tokens.

        Same as `list(self.generate_tokens())`.
        """
        return list(self.generate_tokens())
