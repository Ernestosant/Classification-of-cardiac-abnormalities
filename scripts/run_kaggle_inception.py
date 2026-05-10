from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_inception"
OUTPUT_DIR = ROOT / "kaggle_outputs" / "inception"
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"


def kaggle_env() -> dict[str, str]:
    candidates = [
        ROOT,
        ROOT / ".kaggle",
        Path.home() / ".kaggle",
    ]
    existing = [path for path in candidates if (path / "kaggle.json").exists()]
    if not existing:
        raise SystemExit("No kaggle.json found in repo root, .kaggle/, or the user home .kaggle/ directory.")

    for candidate in existing:
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(candidate)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "1"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode == 0:
            print(f"Using Kaggle config directory: {candidate}")
            return env

    raise SystemExit("Found kaggle.json, but none of the checked credentials authenticated successfully.")


def kernel_id() -> str:
    data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return data["id"]


def run(cmd: list[str], env: dict[str, str]) -> str:
    print("+ " + " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def submit(env: dict[str, str], accelerator: str, timeout: int | None) -> None:
    cmd = ["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)]
    if accelerator != "none":
        cmd.extend(["--accelerator", accelerator])
    if timeout:
        cmd.extend(["--timeout", str(timeout)])
    run(cmd, env)


def wait_until_finished(env: dict[str, str], poll_seconds: int, max_minutes: int) -> None:
    kid = kernel_id()
    deadline = time.time() + max_minutes * 60
    while True:
        out = run(["kaggle", "kernels", "status", kid], env).lower()
        if "complete" in out:
            return
        if any(word in out for word in ["error", "failed", "canceled", "cancelled"]):
            raise SystemExit(f"Kaggle kernel did not complete successfully: {out}")
        if time.time() >= deadline:
            raise SystemExit("Timed out while waiting for Kaggle kernel output.")
        time.sleep(poll_seconds)


def download(env: dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kernel_id(), "-p", str(OUTPUT_DIR), "-o", "-q"], env)


def copy_artifacts() -> None:
    archive = OUTPUT_DIR / "inception_artifacts.zip"
    extracted = OUTPUT_DIR / "extracted"
    if archive.exists():
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        source_root = extracted
    else:
        source_root = OUTPUT_DIR

    for folder in ["models", "reports"]:
        src = source_root / folder
        dst = ROOT / folder
        if not src.exists():
            raise SystemExit(f"Missing expected Kaggle output folder: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)
    print("Copied Kaggle Inception artifacts into local models/ and reports/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and retrieve the Kaggle InceptionTime training kernel.")
    parser.add_argument("--submit", action="store_true", help="Push and start the Kaggle kernel.")
    parser.add_argument("--wait", action="store_true", help="Poll until the kernel finishes.")
    parser.add_argument("--download", action="store_true", help="Download kernel outputs.")
    parser.add_argument("--copy-artifacts", action="store_true", help="Copy downloaded models/reports into the repo.")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "tpu", "none"])
    parser.add_argument("--timeout", type=int, default=None, help="Optional Kaggle run timeout in seconds.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-minutes", type=int, default=180)
    args = parser.parse_args()

    env = kaggle_env()
    if args.submit:
        submit(env, args.accelerator, args.timeout)
    if args.wait:
        wait_until_finished(env, args.poll_seconds, args.max_minutes)
    if args.download:
        download(env)
    if args.copy_artifacts:
        copy_artifacts()

    if not any([args.submit, args.wait, args.download, args.copy_artifacts]):
        print(f"Kernel id: {kernel_id()}")
        print("No action requested. Use --submit --wait --download --copy-artifacts for the full flow.")


if __name__ == "__main__":
    main()
