from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], label: str) -> tuple[bool, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        shell=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, f"{label}: {'PASS' if completed.returncode == 0 else 'FAIL'}\n{output}".strip()


def check_files() -> tuple[bool, str]:
    required = [
        "openenv.yaml",
        "environment.py",
        "grader.py",
        "inference.py",
        "models.py",
        "Dockerfile",
        "requirements.txt",
        "README.md",
        "pyproject.toml",
        "tests/test_environment.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        return False, f"required-files: FAIL\nMissing: {', '.join(missing)}"
    return True, "required-files: PASS"


def check_inference_output() -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "inference.py", "--mock"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        shell=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, f"inference-format: FAIL\n{output}"

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return False, "inference-format: FAIL\nNo output captured."

    for line in lines:
        if not any(line.startswith(prefix) for prefix in ("[START]", "[STEP]", "[END]")):
            return False, f"inference-format: FAIL\nUnexpected line: {line}"

    start_count = sum(line.startswith("[START]") for line in lines)
    end_count = sum(line.startswith("[END]") for line in lines)
    if start_count != 3 or end_count != 3:
        return False, f"inference-format: FAIL\nExpected 3 START and 3 END lines, got {start_count} and {end_count}."

    return True, "inference-format: PASS"


def main() -> int:
    checks: list[tuple[bool, str]] = [check_files()]
    checks.append(
        run(
            [
                sys.executable,
                "-m",
                "py_compile",
                "models.py",
                "environment.py",
                "grader.py",
                "inference.py",
                "app.py",
                "demo.py",
                "server/app.py",
            ],
            "py-compile",
        )
    )
    checks.append(check_inference_output())

    openenv_path = shutil.which("openenv")
    if not openenv_path:
        local_openenv = ROOT / "venv" / "Scripts" / "openenv.exe"
        if local_openenv.exists():
            openenv_path = str(local_openenv)
    if openenv_path:
        checks.append(run([openenv_path, "validate", ".", "--verbose"], "openenv-validate"))
    else:
        checks.append((False, "openenv-validate: FAIL\nopenenv command not found on PATH."))

    all_ok = True
    for success, message in checks:
        all_ok = all_ok and success
        print(message)
        print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
