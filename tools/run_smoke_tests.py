"""Run smoke tests from explicit Python files or directories."""

import subprocess
from pathlib import Path

FOLDERS_TO_TEST = ["components"]


def run_smoke_tests(paths: list[Path]) -> None:
    """Run smoke tests for the given list of paths."""
    for raw_path in paths:
        module_name = str(raw_path).replace("/", ".").replace(".py", "")
        subprocess.run(["python3", "-m", module_name], check=True)
        print(f"PASSED smoketest {module_name}")


def search_for_scripts(paths: list[Path]) -> list[Path]:
    """Search for Python scripts in the given paths."""
    found_scripts = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found_scripts.extend(sorted(path.rglob("*.py")))
        elif path.is_file() and path.suffix == ".py":
            found_scripts.append(path)

    found_scripts = [p for p in found_scripts if "def _smoke_test()" in p.read_text()]

    return found_scripts


if __name__ == "__main__":
    print("Running smoke tests for explicit files")
    run_smoke_tests(search_for_scripts(FOLDERS_TO_TEST))
