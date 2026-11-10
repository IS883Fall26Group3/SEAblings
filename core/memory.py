"""How much memory this app can still use, so an offline model is never loaded into too little.

On Streamlit Community Cloud running out of memory stops the whole app (no Python error to catch),
so we check before loading. Linux only: reads the container limit (cgroup) and the machine's free
memory. Elsewhere (e.g. macOS) it returns None and the check is skipped."""
from pathlib import Path

MB = 1024 * 1024
ROOT = Path("/")


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None  # "max" (no limit) is not a number


def _meminfo_available(root: Path) -> int | None:
    try:
        for line in (root / "proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024  # the file gives kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def _stat(path: Path, key: str) -> int:
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return 0


def _cgroup_headroom(root: Path) -> int | None:
    """Container limit minus what the app already uses (cgroup v2, then v1). Cached files Linux
    can drop when needed (inactive_file) don't count as used, as in Docker's own memory figure."""
    cg = root / "sys/fs/cgroup"
    for limit_file, used_file, stat_file, cache_key in (
            ("memory.max", "memory.current", "memory.stat", "inactive_file"),
            ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes", "memory/memory.stat",
             "total_inactive_file")):
        limit, used = _read_int(cg / limit_file), _read_int(cg / used_file)
        # v1 reports a huge number when there is no limit; treat anything over 1 PB as none
        if limit is not None and used is not None and limit < 2 ** 50:
            used = max(used - _stat(cg / stat_file, cache_key), 0)
            return max(limit - used, 0)
    return None


def available_mb(root: Path = ROOT) -> int | None:
    """Free memory in MB: the smaller of the container headroom and the machine's free memory."""
    values = [v for v in (_cgroup_headroom(root), _meminfo_available(root)) if v is not None]
    return min(values) // MB if values else None


def not_enough_memory(needed_mb: int, root: Path = ROOT) -> int | None:
    """The free MB if it is below needed_mb, else None (enough, or unknown)."""
    free = available_mb(root)
    return free if free is not None and free < needed_mb else None
