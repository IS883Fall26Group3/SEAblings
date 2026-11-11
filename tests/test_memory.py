"""The free-memory check that runs before an offline model is loaded (core/memory.py)."""
import sys
import types

import pytest

from core import memory
from core.llm.base import LLMError

MB = 1024 * 1024


def fake_root(tmp_path, limit=None, used=None, available_kb=None, v1=False):
    """A fake / with the cgroup and /proc/meminfo files Streamlit Cloud's Linux would have."""
    cg = tmp_path / "sys/fs/cgroup"
    if limit is not None:
        folder, lim, cur = ((cg / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes") if v1
                            else (cg, "memory.max", "memory.current"))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / lim).write_text(str(limit))
        (folder / cur).write_text(str(used))
    if available_kb is not None:
        (tmp_path / "proc").mkdir()
        (tmp_path / "proc/meminfo").write_text(f"MemTotal: 99999999 kB\nMemAvailable: {available_kb} kB\n")
    return tmp_path


def test_container_headroom_is_the_limit_minus_what_is_used(tmp_path):
    root = fake_root(tmp_path, limit=2700 * MB, used=900 * MB, available_kb=8_000_000)
    assert memory.available_mb(root) == 1800


def test_droppable_file_cache_does_not_count_as_used(tmp_path):
    root = fake_root(tmp_path, limit=2700 * MB, used=2000 * MB)
    (tmp_path / "sys/fs/cgroup/memory.stat").write_text(f"anon 1\ninactive_file {800 * MB}\nactive_file 5\n")
    assert memory.available_mb(root) == 1500


def test_cgroup_v1_is_read_too(tmp_path):
    root = fake_root(tmp_path, limit=1000 * MB, used=600 * MB, v1=True)
    assert memory.available_mb(root) == 400


def test_no_container_limit_uses_the_machine_free_memory(tmp_path):
    root = fake_root(tmp_path, available_kb=500 * 1024)
    (tmp_path / "sys/fs/cgroup").mkdir(parents=True)
    (tmp_path / "sys/fs/cgroup/memory.max").write_text("max")
    (tmp_path / "sys/fs/cgroup/memory.current").write_text(str(100 * MB))
    assert memory.available_mb(root) == 500


def test_unknown_memory_never_blocks(tmp_path):
    assert memory.available_mb(tmp_path) is None  # e.g. macOS: no such files
    assert memory.not_enough_memory(1500, tmp_path) is None


def test_not_enough_memory_reports_the_free_amount(tmp_path):
    root = fake_root(tmp_path, limit=2000 * MB, used=1000 * MB)
    assert memory.not_enough_memory(1500, root) == 1000
    assert memory.not_enough_memory(800, root) is None


@pytest.fixture
def low_memory(monkeypatch):
    """llama_cpp 'installed' but only 300 MB free; fails the test if a model load is attempted."""
    class Llama:
        @staticmethod
        def from_pretrained(*a, **k):
            raise AssertionError("the model must not be loaded")
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Llama))
    monkeypatch.setattr(memory, "available_mb", lambda root=memory.ROOT: 300)


def test_offline_llm_is_not_loaded_without_enough_memory(low_memory):
    from core.llm.local_llama import LlamaCppProvider
    with pytest.raises(LLMError, match="needs about 1,500 MB of free memory, but this server has 300 MB") as e:
        LlamaCppProvider("org/Model-GGUF", "*.gguf", min_free_mb=1500)
    assert "Pick a Gemini model" in str(e.value)


def test_offline_embedder_is_not_loaded_without_enough_memory(low_memory):
    from core.embeddings import LocalEmbedder
    with pytest.raises(LLMError, match="similarity model needs about 400 MB"):
        LocalEmbedder("org/bge-gguf", "*.gguf", min_free_mb=400)


def test_offline_embedder_load_failure_is_a_clear_error(monkeypatch):
    class Llama:
        @staticmethod
        def from_pretrained(*a, **k):
            raise ValueError("Failed to create llama_context")
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Llama))
    from core.embeddings import LocalEmbedder
    with pytest.raises(LLMError, match="Couldn't load the offline similarity model org/bge-gguf"):
        LocalEmbedder("org/bge-gguf", "*.gguf")
