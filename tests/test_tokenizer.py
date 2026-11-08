"""Tests for the tokenizer.

Run from the project folder:
    .venv\\Scripts\\python.exe -m pytest tests
"""

import re

import pytest

from langpy.errors import LanguageError
from langpy.tokenizer import (
    InvalidTokenError,
    Token,
    Tokenizer,
    TokenizerError,
    ValidToken,
    ValidTokenRule,
)

exact = ValidToken.exact
regex = ValidToken.regex

# A small but realistic language: keywords, identifiers, numbers,
# strings with escapes, line comments, operators and punctuation.
LANGUAGE = [
    *(exact(k) for k in ["let", "if", "else", "while", "fn", "return"]),
    *(exact(op) for op in ["=", "==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/"]),
    *(exact(p) for p in ["(", ")", "{", "}", ",", ";"]),
    regex(r"[A-Za-z_][A-Za-z0-9_]*", name="IDENT"),
    regex(r"[0-9]+(\.[0-9]+)?", name="NUMBER"),
    regex(r'"(\\.|[^"\\])*"', name="STRING"),
    regex(r"#[^\n]*", name="COMMENT"),
]


def values(source: str, rules: list[ValidToken] = LANGUAGE, **kwargs) -> list[str]:
    return [token.value for token in Tokenizer(source, rules, **kwargs).run()]


def types(source: str, rules: list[ValidToken] = LANGUAGE, **kwargs) -> list[str]:
    return [token.type for token in Tokenizer(source, rules, **kwargs).run()]


# --- Rules ------------------------------------------------------------------


def test_rule_name_defaults_to_pattern():
    assert exact("==").name == "=="
    assert regex("[0-9]+").name == "[0-9]+"
    assert regex("[0-9]+", name="NUMBER").name == "NUMBER"


def test_helpers_match_constructor():
    assert exact("let") == ValidToken(ValidTokenRule.EXACT, "let")
    assert regex("a+") == ValidToken(ValidTokenRule.REGEX, "a+")


def test_empty_pattern_is_rejected():
    with pytest.raises(ValueError):
        exact("")


def test_invalid_regex_is_rejected_early():
    with pytest.raises(re.error):
        regex("[unclosed")


def test_regex_flags():
    rules = [regex("select", name="SELECT", flags=re.IGNORECASE)]
    assert types("select SELECT Select", rules) == ["SELECT"] * 3


# --- Matching ---------------------------------------------------------------


def test_exact_tokens():
    assert values("let x = 1 + 2;") == ["let", "x", "=", "1", "+", "2", ";"]


def test_token_types():
    assert types('let x = 1.5 + "s"; # c') == [
        "let", "IDENT", "=", "NUMBER", "+", "STRING", ";", "COMMENT",
    ]  # fmt: skip


@pytest.mark.parametrize(
    "source, expected",
    [
        ("a==b", ["a", "==", "b"]),
        ("a===b", ["a", "==", "=", "b"]),
        ("a<=b", ["a", "<=", "b"]),
        ("a!=b", ["a", "!=", "b"]),
    ],
)
def test_longest_exact_match_wins(source, expected):
    assert values(source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("let", ["let"]),
        ("letter", ["IDENT"]),  # identifier that starts with a keyword
        ("iffy", ["IDENT"]),
        ("let1", ["IDENT"]),
        ("fn(x)", ["fn", "(", "IDENT", ")"]),
    ],
)
def test_keywords_vs_identifiers(source, expected):
    assert types(source) == expected


def test_exact_beats_regex_on_tie_regardless_of_order():
    rules = [regex(r"[a-z]+", name="IDENT"), exact("let")]
    assert types("let lets", rules) == ["let", "IDENT"]


def test_first_rule_wins_tie_between_regexes():
    rules = [regex(r"[a-z]+", name="A"), regex(r"[a-z]+", name="B")]
    assert types("abc", rules) == ["A"]


def test_regex_only_matches_at_current_position():
    # The digit rule must not skip past "a" to find "1" later on.
    with pytest.raises(InvalidTokenError) as err:
        values("a1", [regex(r"[0-9]+")])
    assert (err.value.line, err.value.column) == (1, 1)


def test_zero_length_regex_match_is_ignored():
    rules = [regex(r"[0-9]*", name="DIGITS"), exact("x")]
    assert values("x12x", rules) == ["x", "12", "x"]


def test_only_zero_length_match_is_an_error():
    with pytest.raises(InvalidTokenError):
        values("x", [regex(r"[0-9]*")])


def test_number_followed_by_operator():
    assert values("1.5*2-3") == ["1.5", "*", "2", "-", "3"]


# --- Strings and comments ---------------------------------------------------


def test_string_keeps_inner_whitespace():
    assert values('x = "hello   world"') == ["x", "=", '"hello   world"']


def test_string_with_escaped_quote():
    assert values(r'"say \"hi\"" ;') == [r'"say \"hi\""', ";"]


def test_string_containing_other_tokens():
    assert values('"let x = 1 # not a comment"') == ['"let x = 1 # not a comment"']


def test_comment_runs_to_end_of_line():
    assert values("x = 1 # set x\ny = 2") == ["x", "=", "1", "# set x", "y", "=", "2"]


def test_skip_rule_hides_matches():
    rules = LANGUAGE[:-1] + [regex(r"#[^\n]*", skip=True)]
    assert values("# header\nx = 1 # set x\ny", rules) == ["x", "=", "1", "y"]


# --- Whitespace -------------------------------------------------------------


@pytest.mark.parametrize("source", ["", " ", "\n\n", " \t\r\n "])
def test_empty_or_whitespace_only(source):
    assert values(source) == []


def test_whitespace_between_and_around_tokens():
    assert values("\n\t let\n  x\r\n=\t1  \n") == ["let", "x", "=", "1"]


def test_lots_of_whitespace_does_not_crash():
    source = "x" + " " * 50_000 + "\n" * 50_000 + "y"
    assert values(source) == ["x", "y"]


def test_whitespace_can_be_tokens():
    rules = [
        regex(r"[a-z]+", name="IDENT"),
        regex(r"\n", name="NEWLINE"),
        regex(r"[ \t]+", name="INDENT"),
    ]
    source = "if\n    x\ny"
    assert types(source, rules, skip_whitespace=False) == [
        "IDENT", "NEWLINE", "INDENT", "IDENT", "NEWLINE", "IDENT",
    ]  # fmt: skip


def test_unhandled_whitespace_is_an_error_when_not_skipped():
    with pytest.raises(InvalidTokenError):
        values("a b", [regex("[a-z]")], skip_whitespace=False)


# --- Positions --------------------------------------------------------------


def test_token_positions():
    tokens = Tokenizer("let x\n  = 10", LANGUAGE).run()
    assert tokens == [
        Token("let", "let", 1, 1, 0),
        Token("IDENT", "x", 1, 5, 4),
        Token("=", "=", 2, 3, 8),
        Token("NUMBER", "10", 2, 5, 10),
    ]


def test_positions_after_multi_line_token():
    rules = LANGUAGE + [regex(r'"""(.|\n)*?"""', name="DOC")]
    tokens = Tokenizer('"""a\nbc""" x', rules).run()
    assert (tokens[1].line, tokens[1].column) == (2, 7)


# --- Full program -----------------------------------------------------------


def test_small_program():
    source = """\
# compute a factorial
fn fact(n) {
    if n <= 1 { return 1; }
    return n * fact(n - 1);
}
let result = fact(5);
let msg = "done";
"""
    assert values(source) == [
        "# compute a factorial",
        "fn", "fact", "(", "n", ")", "{",
        "if", "n", "<=", "1", "{", "return", "1", ";", "}",
        "return", "n", "*", "fact", "(", "n", "-", "1", ")", ";",
        "}",
        "let", "result", "=", "fact", "(", "5", ")", ";",
        "let", "msg", "=", '"done"', ";",
    ]  # fmt: skip


# --- Errors -----------------------------------------------------------------


def test_invalid_token_reports_line_and_column():
    with pytest.raises(InvalidTokenError) as err:
        values("let x = 1;\n  y = $")
    assert (err.value.line, err.value.column) == (2, 7)
    assert "'$'" in str(err.value)


def test_unterminated_string_is_invalid():
    with pytest.raises(InvalidTokenError):
        values('x = "oops')


def test_error_hierarchy():
    with pytest.raises(TokenizerError):
        values("@")
    with pytest.raises(LanguageError):
        values("@")


def test_tokens_before_error_are_returned():
    tokenizer = Tokenizer("x + @", LANGUAGE)
    assert tokenizer.next_token().value == "x"
    assert tokenizer.next_token().value == "+"
    with pytest.raises(InvalidTokenError):
        tokenizer.next_token()


# --- Tokenizer API ----------------------------------------------------------


def test_iteration_and_generator():
    assert [t.value for t in Tokenizer("x + y", LANGUAGE)] == ["x", "+", "y"]
    assert [t.value for t in Tokenizer("x + y", LANGUAGE).generate_tokens()] == ["x", "+", "y"]


def test_next_token_returns_none_at_end():
    tokenizer = Tokenizer("x", LANGUAGE)
    assert tokenizer.next_token().value == "x"
    assert tokenizer.next_token() is None
    assert tokenizer.next_token() is None


def test_peek_does_not_advance():
    tokenizer = Tokenizer("a + b", LANGUAGE)
    assert tokenizer.peek_token().value == "a"
    assert tokenizer.peek_token(1).value == "+"
    assert tokenizer.peek_token(2).value == "b"
    assert tokenizer.peek_token(3) is None
    assert tokenizer.next_token().value == "a"
    assert tokenizer.peek_token().value == "+"
    assert tokenizer.get_current_token().value == "a"


def test_peek_restores_position_after_error():
    tokenizer = Tokenizer("a @", LANGUAGE)
    with pytest.raises(InvalidTokenError):
        tokenizer.peek_token(1)
    assert tokenizer.next_token().value == "a"


def test_get_current_token():
    tokenizer = Tokenizer("let count", LANGUAGE)
    assert tokenizer.get_current_token() is None
    tokenizer.next_token()
    assert tokenizer.get_current_token().value == "let"
    tokenizer.next_token()
    assert tokenizer.get_current_token().value == "count"


def test_line_and_column_properties():
    tokenizer = Tokenizer("ab\ncd", LANGUAGE)
    tokenizer.run()
    assert (tokenizer.line, tokenizer.column) == (2, 3)


def test_reset():
    tokenizer = Tokenizer("x + y", LANGUAGE)
    first = tokenizer.run()
    tokenizer.reset()
    assert tokenizer.run() == first


def test_set_source_code():
    tokenizer = Tokenizer("x", LANGUAGE)
    tokenizer.run()
    tokenizer.set_source_code("y + 1")
    assert [t.value for t in tokenizer.run()] == ["y", "+", "1"]
