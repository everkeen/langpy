"""Errors"""


class LanguageError(Exception):
    """An error related to the input text.

    line, column: Where the error happened (both start at 1), or None if unknown.
    """

    message: str
    line: int | None
    column: int | None

    def __init__(self, message: str, line: int | None = None, column: int | None = None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        if self.line is None:
            return self.message
        return f"{self.message} (line {self.line}, column {self.column})"

    def format(self, source: str | None = None) -> str:
        """Returns a readable report, showing the offending line of `source` if given.

        Example output:
            Line 2, column 7: Unexpected character '$'.
                y = $
                    ^
        """
        if self.line is None:
            return f"{type(self).__name__}: {self.message}"
        report = f"Line {self.line}, column {self.column}: {self.message}"
        lines = source.splitlines() if source is not None else []
        if 1 <= self.line <= len(lines):
            text = lines[self.line - 1]
            caret = " " * (max(self.column or 1, 1) - 1) + "^"
            report += f"\n    {text}\n    {caret}"
        return report
