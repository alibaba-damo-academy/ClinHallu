#!/usr/bin/env python3
"""Batch-download model caches via the ModelScope CLI."""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELSCOPE_BIN = "modelscope"


@dataclass
class DownloadTask:
    name: str
    repo_id: Optional[str]
    model_id: Optional[str]
    cache_dir: Path
    mode: str = "download"
    revision: str = "main"
    source_dir: Optional[Path] = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model caches defined in configs/config.yaml via ModelScope"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the unified config file.",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="model_downloads",
        help="Config section that contains the download manifest.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated model names, repo_ids, or model_ids to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the state file and rerun all matching tasks.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Deprecated compatibility flag from hfd.sh. Ignored by ModelScope downloads.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Deprecated compatibility flag from hfd.sh. Ignored by ModelScope downloads.",
    )
    parser.add_argument(
        "--ms-token",
        type=str,
        default="",
        help="Optional ModelScope access token.",
    )
    parser.add_argument(
        "--hf-username",
        type=str,
        default="",
        help="Deprecated compatibility flag from hfd.sh. Ignored by ModelScope downloads.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Deprecated alias for --ms-token.",
    )
    parser.add_argument(
        "--modelscope-bin",
        type=str,
        default=DEFAULT_MODELSCOPE_BIN,
        help="Path to the ModelScope CLI executable.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle) or {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _as_path(value: Optional[str], default: Optional[Path] = None) -> Path:
    if value:
        return Path(value).expanduser()
    if default is not None:
        return default
    return Path(".")


def _positive_int(value: Optional[int], default: int) -> int:
    resolved = default if value is None else value
    if resolved < 1:
        raise ValueError("Expected a positive integer")
    return resolved


def _parse_task_list(section: Dict[str, Any], fallback_root: Path) -> List[DownloadTask]:
    tasks: List[DownloadTask] = []
    for raw_item in section.get("models", []):
        if not isinstance(raw_item, dict):
            continue
        repo_id = raw_item.get("repo_id")
        model_id = raw_item.get("model_id") or repo_id
        name = str(raw_item.get("name") or model_id or repo_id or "").strip()
        cache_dir = _as_path(raw_item.get("cache_dir"), fallback_root / name)
        source_dir = raw_item.get("source_dir")
        tasks.append(
            DownloadTask(
                name=name or cache_dir.name,
                repo_id=repo_id,
                model_id=str(model_id).strip() if model_id else None,
                cache_dir=cache_dir,
                mode=str(raw_item.get("mode", "download")).strip().lower(),
                revision=str(raw_item.get("revision", "main")).strip(),
                source_dir=_as_path(source_dir) if source_dir else None,
            )
        )
    return tasks


def _filter_tasks(tasks: List[DownloadTask], only: Iterable[str]) -> List[DownloadTask]:
    selected = {item.strip() for item in only if item.strip()}
    if not selected:
        return tasks
    return [
        task
        for task in tasks
        if task.name in selected
        or (task.repo_id and task.repo_id in selected)
        or (task.model_id and task.model_id in selected)
    ]


def _cache_has_content(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _load_state(state_path: Path) -> Dict[str, Any]:
    state = _load_json(state_path)
    state.setdefault("completed", {})
    state.setdefault("failed", {})
    return state


def _mark_completed(state: Dict[str, Any], task: DownloadTask, details: Dict[str, Any]) -> None:
    state["completed"][task.name] = {
        "repo_id": task.repo_id,
        "model_id": task.model_id,
        "cache_dir": str(task.cache_dir),
        "mode": task.mode,
        "revision": task.revision,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **details,
    }
    state["failed"].pop(task.name, None)


def _mark_failed(state: Dict[str, Any], task: DownloadTask, error: str) -> None:
    state["failed"][task.name] = {
        "repo_id": task.repo_id,
        "model_id": task.model_id,
        "cache_dir": str(task.cache_dir),
        "mode": task.mode,
        "revision": task.revision,
        "error": error,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _sync_local_directory(source_dir: Path, cache_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source cache not found: {source_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        subprocess.run(
            [rsync, "-a", "--partial", "--inplace", f"{source_dir}/", f"{cache_dir}/"],
            check=True,
        )
        return

    shutil.copytree(source_dir, cache_dir, symlinks=True, dirs_exist_ok=True)


def _resolve_modelscope_bin(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"ModelScope CLI not found: {candidate}")

    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise FileNotFoundError(
        "ModelScope CLI not found in PATH. Install it first or pass --modelscope-bin."
    )


def _run_modelscope_download(
    task: DownloadTask,
    modelscope_bin: str,
    token: Optional[str],
    attempts: int,
    retry_sleep_seconds: int,
) -> Dict[str, Any]:
    if not task.model_id:
        raise ValueError(
            f"Missing model_id/repo_id for download task '{task.name}'"
        )

    env = os.environ.copy()
    if token:
        env.setdefault("MODELSCOPE_API_TOKEN", token)
        env.setdefault("MODELSCOPE_TOKEN", token)

    command = [
        modelscope_bin,
        "download",
        "--model",
        task.model_id,
        "--local_dir",
        str(task.cache_dir),
    ]
    if task.revision and task.revision != "main":
        command.extend(["--revision", task.revision])

    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            logger.info(
                "modelscope download %s (attempt %s/%s): %s",
                task.name,
                attempt,
                attempts,
                " ".join(command),
            )
            subprocess.run(command, check=True, cwd=str(REPO_ROOT), env=env)
            return {
                "mode": "download",
                "repo_id": task.repo_id,
                "model_id": task.model_id,
                "cache_dir": str(task.cache_dir),
                "modelscope_bin": modelscope_bin,
            }
        except KeyboardInterrupt:
            raise
        except subprocess.CalledProcessError as error:
            last_error = error
            logger.warning(
                "Download failed for %s (attempt %s/%s): return code %s",
                task.name,
                attempt,
                attempts,
                error.returncode,
            )
            if attempt < attempts:
                time.sleep(min(retry_sleep_seconds * attempt, 300))

    raise RuntimeError(f"Download failed for {task.name}") from last_error


def _run_task(
    task: DownloadTask,
    modelscope_bin: str,
    token: Optional[str],
    attempts: int,
    retry_sleep_seconds: int,
) -> Dict[str, Any]:
    task.cache_dir.mkdir(parents=True, exist_ok=True)

    if task.mode == "copy":
        if task.source_dir is None:
            raise ValueError(f"Task '{task.name}' is copy mode but source_dir is missing")
        _sync_local_directory(task.source_dir, task.cache_dir)
        return {
            "mode": "copy",
            "source_dir": str(task.source_dir),
            "cache_dir": str(task.cache_dir),
        }

    if task.mode != "download":
        raise ValueError(f"Unsupported task mode '{task.mode}' for '{task.name}'")

    return _run_modelscope_download(
        task=task,
        modelscope_bin=modelscope_bin,
        token=token,
        attempts=attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )


def main() -> int:
    args = parse_arguments()
    config = load_config(args.config)
    section = config.get(args.section, {})
    if not isinstance(section, dict) or not section:
        raise ValueError(f"Missing download section '{args.section}' in {args.config}")

    cache_root = _as_path(
        section.get("cache_root"),
        _as_path(config.get("paths", {}).get("model_cache_root"), Path("model_cache")),
    )
    max_rounds = int(section.get("max_rounds", 12))
    retry_sleep_seconds = int(section.get("retry_sleep_seconds", 30))
    per_model_attempts = int(section.get("per_model_attempts", 4))
    tasks = _parse_task_list(section, cache_root)
    if not tasks:
        raise ValueError(f"No models listed in '{args.section}.models'")

    if args.only:
        tasks = _filter_tasks(tasks, args.only.split(","))
    if not tasks:
        raise ValueError("No download tasks matched the requested subset")

    state_path = _as_path(section.get("state_file"), cache_root / ".download_state.json")
    state = _load_state(state_path)
    if args.force:
        state = {"completed": {}, "failed": {}}

    pending = []
    for task in tasks:
        completed_entry = state["completed"].get(task.name)
        if completed_entry and _cache_has_content(task.cache_dir) and not args.force:
            logger.info("Skipping completed task: %s", task.name)
            continue
        pending.append(task)

    if args.threads is not None or "threads" in section:
        logger.info(
            "Ignoring legacy --threads/hfd.sh setting: %s",
            _positive_int(args.threads, int(section.get("threads", 10))),
        )
    if args.jobs is not None or "jobs" in section:
        logger.info(
            "Ignoring legacy --jobs/hfd.sh setting: %s",
            _positive_int(args.jobs, int(section.get("jobs", 10))),
        )
    if section.get("endpoint"):
        logger.info("Ignoring legacy HF endpoint setting: %s", section.get("endpoint"))
    if args.hf_username or section.get("username"):
        logger.info("Ignoring legacy HF username setting for ModelScope downloads.")

    if not pending:
        logger.info("All requested models are already present.")
        return 0

    modelscope_bin: Optional[str] = None
    if any(task.mode == "download" for task in pending):
        modelscope_bin = _resolve_modelscope_bin(args.modelscope_bin)

    token = (
        args.ms_token
        or args.hf_token
        or section.get("token")
        or os.environ.get("MODELSCOPE_API_TOKEN")
        or os.environ.get("MODELSCOPE_TOKEN")
    )

    logger.info("Cache root: %s", cache_root)
    if modelscope_bin:
        logger.info("ModelScope CLI: %s", modelscope_bin)
    logger.info("State file: %s", state_path)
    logger.info("Pending tasks: %s", [task.name for task in pending])

    for round_index in range(1, max_rounds + 1):
        if not pending:
            break

        logger.info("Download round %s/%s", round_index, max_rounds)
        next_pending: List[DownloadTask] = []
        progress = False

        for task in pending:
            logger.info("Processing %s -> %s", task.name, task.cache_dir)
            try:
                details = _run_task(
                    task=task,
                    modelscope_bin=modelscope_bin or args.modelscope_bin,
                    token=token,
                    attempts=per_model_attempts,
                    retry_sleep_seconds=retry_sleep_seconds,
                )
                _mark_completed(state, task, details)
                _atomic_write_json(state_path, state)
                progress = True
                logger.info("Done: %s", task.name)
            except KeyboardInterrupt:
                _mark_failed(state, task, "Interrupted by user")
                _atomic_write_json(state_path, state)
                raise
            except Exception as error:
                logger.error("Failed: %s | %s", task.name, error)
                _mark_failed(state, task, str(error))
                _atomic_write_json(state_path, state)
                next_pending.append(task)

        pending = next_pending
        if pending and round_index < max_rounds:
            if not progress:
                logger.info("No progress in this round; sleeping before retry.")
            time.sleep(retry_sleep_seconds)

    if pending:
        logger.error(
            "Still pending after %s rounds: %s",
            max_rounds,
            [task.name for task in pending],
        )
        return 1

    logger.info("All downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
