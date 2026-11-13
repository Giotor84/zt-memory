from zt_memory.outline import Outline, outline, short_name

PY = '''@dataclass
class Pricing:
    """Prices with VAT.

    More text."""

    def apply_discount(self, price: float,
                       percent: float) -> float:  # note
        """Price after a percentage discount."""
        return round(price - price * percent, 2)
'''


def test_python_class_and_multiline_method():
    assert outline(PY, 10, ["Pricing", "Pricing.apply_discount"]) == [
        Outline(11, "class Pricing:", "Prices with VAT."),
        Outline(16, "def apply_discount(self, price: float, percent: float) -> float:", "Price after a percentage discount."),
    ]


def test_brace_languages_and_comment_docs():
    ts = '/**\n * Parses a size like "10kb".\n */\nexport function parseSize(text: string): number\n{\n  return 1;\n}\n'
    assert outline(ts, 1, ["parseSize"]) == [Outline(4, "export function parseSize(text: string): number", 'Parses a size like "10kb".')]
    go = "// Execute runs the command.\nfunc (c *Command) Execute() error {\n\treturn nil\n}\n"
    assert outline(go, 40, ["Command.Execute"]) == [Outline(41, "func (c *Command) Execute() error", "Execute runs the command.")]
    rs = "/// Adds one.\n#[inline]\npub fn add_one(x: i32) -> i32 {\n    x + 1\n}\n"
    assert outline(rs, 1, ["add_one~ab12"]) == [Outline(3, "pub fn add_one(x: i32) -> i32", "Adds one.")]
    arrow = "const toBytes = (n: number) => {\n  return n;\n};\n"
    assert outline(arrow, 5, ["toBytes"])[0].signature == "const toBytes = (n: number) =>"


def test_non_symbol_chunks_fall_back_to_first_code_line():
    assert short_name("<gap:12>") is None and short_name("A.b~x") == "b"
    assert outline("# comment\nimport os\nX = 1\n", 1, ["<header>"]) == [Outline(2, "import os")]
