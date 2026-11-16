"""Install persistent publication guards that survive checking out old branches."""
from pathlib import Path
import shutil
import subprocess


def main():
    source = Path(__file__).resolve().parents[1]
    target = Path(subprocess.check_output(["git", "rev-parse", "--git-path", "publication-hooks"], text=True).strip()).resolve()
    current = subprocess.run(["git", "config", "--get", "core.hooksPath"], capture_output=True, text=True).stdout.strip()
    if current not in ("", ".githooks", str(target)):
        raise SystemExit("Existing custom hooks are configured; integrate publication guards before replacing them.")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("pre-commit", "pre-push"):
        shutil.copyfile(source / ".githooks" / name, target / name)
        (target / name).chmod(0o755)
    shutil.copyfile(source / "scripts/check_publication.py", target / "check_publication.py")
    subprocess.run(["git", "config", "core.hooksPath", str(target)], check=True)
    print("Persistent publication guards installed: " + str(target))


if __name__ == "__main__":
    main()
