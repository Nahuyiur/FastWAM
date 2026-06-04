from __future__ import annotations

from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


class LMDBEpisodeStore:
    def __init__(self, data_dir: str | Path, *, readahead: bool = False, max_readers: int = 2048):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.readahead = bool(readahead)
        self.max_readers = int(max_readers)
        self._envs: dict[str, lmdb.Environment] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_envs"] = {}
        return state

    def _taskvar_path(self, taskvar: str) -> Path:
        return self.data_dir / taskvar

    def has_taskvar(self, taskvar: str) -> bool:
        path = self._taskvar_path(taskvar)
        return path.is_dir() and (path / "data.mdb").is_file() and (path / "results.json").is_file()

    def list_taskvars(self) -> list[str]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"GEMBench LMDB directory not found: {self.data_dir}")
        taskvars = []
        for path in sorted(self.data_dir.iterdir()):
            if path.is_dir() and self.has_taskvar(path.name):
                taskvars.append(path.name)
        return taskvars

    def _open_env(self, taskvar: str) -> lmdb.Environment:
        path = self._taskvar_path(taskvar)
        if not path.is_dir():
            raise FileNotFoundError(f"GEMBench taskvar directory not found: {path}")
        if not (path / "data.mdb").is_file():
            raise FileNotFoundError(f"GEMBench LMDB data.mdb not found: {path / 'data.mdb'}")
        return lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            readahead=self.readahead,
            max_readers=self.max_readers,
            subdir=True,
        )

    def _env(self, taskvar: str) -> lmdb.Environment:
        env = self._envs.get(taskvar)
        if env is None:
            env = self._open_env(taskvar)
            self._envs[taskvar] = env
        return env

    def list_episode_keys(self, taskvar: str) -> list[bytes]:
        env = self._open_env(taskvar)
        try:
            with env.begin(write=False) as txn:
                keys = list(txn.cursor().iternext(values=False))
        finally:
            env.close()
        return sorted(keys, key=_episode_sort_key)

    def get(self, taskvar: str, key: bytes | str) -> dict:
        key_b = key.encode("ascii") if isinstance(key, str) else key
        with self._env(taskvar).begin(write=False) as txn:
            value = txn.get(key_b)
        if value is None:
            raise KeyError(f"Missing GEMBench episode key {key_b!r} for taskvar {taskvar!r}")
        return msgpack.unpackb(value, raw=False)

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()


def _episode_sort_key(key: bytes) -> tuple[int, str]:
    text = key.decode("ascii", errors="ignore")
    if text.startswith("episode"):
        suffix = text[len("episode"):]
        if suffix.isdigit():
            return int(suffix), text
    return 10**12, text
