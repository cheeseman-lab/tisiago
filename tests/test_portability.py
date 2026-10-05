from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "path",
    [
        *Path("configs").glob("*.yaml"),
        *Path("src").rglob("*.py"),
    ],
)
def test_library_files_do_not_embed_local_user_paths(path):
    """Site paths belong in .env, not in library code or backend configs."""
    text = path.read_text()
    forbidden = ("/" + "lab/", "/" + "home/", "mdi" + "berna", "test_" + "matteo")
    found = [token for token in forbidden if token in text]
    assert not found, f"{path} contains machine-specific tokens: {found}"
