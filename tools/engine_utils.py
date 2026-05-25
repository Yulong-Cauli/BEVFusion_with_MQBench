import os
import re
from typing import Callable, List, Tuple

import torch


_SM_RE = re.compile(r"^(?P<prefix>.+)_sm(?P<sm>\d{2})(?P<suffix>\.engine)$")


def current_sm_tag() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot resolve TRT engine architecture.")
    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    return f"{prop.major}{prop.minor}"


def _replace_sm_suffix(path: str, sm_tag: str) -> str:
    dirname, basename = os.path.split(path)
    m = _SM_RE.match(basename)
    if not m:
        return path
    replaced = f"{m.group('prefix')}_sm{sm_tag}{m.group('suffix')}"
    return os.path.join(dirname, replaced)


def _sm_suffix(path: str) -> str:
    m = _SM_RE.match(os.path.basename(path))
    return m.group("sm") if m else ""


def candidate_engine_paths(requested_path: str, sm_tag: str) -> List[str]:
    requested_path = requested_path.strip()
    dirname, basename = os.path.split(requested_path)
    paths: List[str] = []

    def add(path: str):
        if path and path not in paths:
            paths.append(path)

    # As requested
    add(requested_path)
    add(_replace_sm_suffix(requested_path, sm_tag))

    # If no SM suffix in name, prefer an SM-matched sibling if present.
    if not _sm_suffix(requested_path):
        stem, ext = os.path.splitext(basename)
        if ext == ".engine":
            sm_name = f"{stem}_sm{sm_tag}.engine"
            if dirname:
                add(os.path.join(dirname, sm_name))
            else:
                add(sm_name)

    return paths


def load_runner_with_fallback(
    requested_path: str,
    runner_ctor: Callable[[str, object], object],
    logger,
    role: str,
) -> Tuple[str, object]:
    requested_path = requested_path.strip()
    if os.path.dirname(requested_path) == "artifacts":
        normalized = os.path.basename(requested_path)
        logger.warning(
            f"[{role}] artifacts path is archive-only, use root engine instead: "
            f"{requested_path} -> {normalized}"
        )
        requested_path = normalized
    sm_tag = current_sm_tag()
    candidates = candidate_engine_paths(requested_path, sm_tag)
    attempted = []

    for path in candidates:
        if not os.path.exists(path):
            attempted.append((path, "missing"))
            continue
        try:
            runner = runner_ctor(path, logger)
            if path != requested_path:
                logger.warning(
                    f"[{role}] engine fallback: {requested_path} -> {path} (sm{sm_tag})"
                )
            return path, runner
        except Exception as exc:  # keep trying alternatives
            attempted.append((path, str(exc)))

    detail = "\n".join([f"  - {p}: {msg}" for p, msg in attempted])
    raise RuntimeError(
        f"[{role}] No compatible TRT engine for sm{sm_tag}. Requested: {requested_path}\n"
        f"Tried:\n{detail}"
    )
