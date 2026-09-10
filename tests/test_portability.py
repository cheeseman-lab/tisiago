from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "path",
    [
        Path("README.md"),
        Path("CLAUDE.md"),
        Path("HANDOFF_OPTION_B.md"),
        *Path("docs").rglob("*.md"),
        *Path("configs").glob("*.yaml"),
        *Path("scripts").glob("*.sh"),
        *Path("scripts").glob("*.py"),
        *Path("autoresearch").glob("*.sh"),
        *Path("autoresearch").glob("*.py"),
        *Path("src").rglob("*.py"),
    ],
)
def test_runtime_files_do_not_embed_local_user_paths(path):
    """Keep executable examples and current docs usable outside the author's cluster."""
    text = path.read_text()
    forbidden = ("/" + "lab/", "/" + "home/", "mdi" + "berna", "test_" + "matteo")
    found = [token for token in forbidden if token in text]
    assert not found, f"{path} contains machine-specific tokens: {found}"
