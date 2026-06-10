from __future__ import annotations

from pathlib import Path
from typing import Any

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

    def key_frameids(self, taskvar: str, key: bytes | str) -> list[int]:
        """Read `key_frameids` without materializing RGB/depth/point-cloud arrays."""
        key_b = key.encode("ascii") if isinstance(key, str) else key
        with self._env(taskvar).begin(write=False, buffers=True) as txn:
            value = txn.get(key_b)
            if value is None:
                raise KeyError(f"Missing GEMBench episode key {key_b!r} for taskvar {taskvar!r}")
            try:
                return _msgpack_top_level_int_array(value, b"key_frameids")
            except Exception:
                episode = msgpack.unpackb(bytes(value), raw=False)
        key_frameids = episode.get("key_frameids")
        if key_frameids is None:
            raise KeyError(f"Missing `key_frameids` in GEMBench episode {taskvar}/{key_b!r}")
        return [int(value) for value in key_frameids]

    def key_frame_count(self, taskvar: str, key: bytes | str) -> int:
        return len(self.key_frameids(taskvar, key))

    def array_shape(self, taskvar: str, key: bytes | str, field: str) -> tuple[int, ...]:
        return self.array_shapes(taskvar, key, (field,))[field]

    def array_shapes(self, taskvar: str, key: bytes | str, fields: list[str] | tuple[str, ...]) -> dict[str, tuple[int, ...]]:
        key_b = key.encode("ascii") if isinstance(key, str) else key
        field_names = [str(field) for field in fields]
        field_map = {name.encode("utf-8"): name for name in field_names}
        with self._env(taskvar).begin(write=False, buffers=True) as txn:
            value = txn.get(key_b)
            if value is None:
                raise KeyError(f"Missing GEMBench episode key {key_b!r} for taskvar {taskvar!r}")
            try:
                by_key = _msgpack_top_level_numpy_shapes(value, tuple(field_map.keys()))
                return {field_map[key]: shape for key, shape in by_key.items()}
            except Exception:
                episode = msgpack.unpackb(bytes(value), raw=False)
        out: dict[str, tuple[int, ...]] = {}
        for field in field_names:
            array = episode.get(field)
            if array is None:
                raise KeyError(f"Missing `{field}` in GEMBench episode {taskvar}/{key_b!r}")
            shape = getattr(array, "shape", None)
            if shape is None and isinstance(array, dict):
                shape = array.get(b"shape") or array.get("shape")
            if shape is None:
                raise ValueError(f"Could not determine `{field}` shape for GEMBench episode {taskvar}/{key_b!r}")
            out[field] = tuple(int(dim) for dim in shape)
        return out

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


def _byte(buf: Any, pos: int) -> int:
    value = buf[pos]
    return int(value) if not isinstance(value, bytes) else value[0]


def _read_uint(buf: Any, pos: int, nbytes: int) -> tuple[int, int]:
    end = pos + nbytes
    return int.from_bytes(buf[pos:end], byteorder="big", signed=False), end


def _read_len(buf: Any, pos: int, code: int, *, base: int, fixed_mask: int | None = None) -> tuple[int, int]:
    if fixed_mask is not None:
        return code & fixed_mask, pos
    if code == base:
        return _read_uint(buf, pos, 1)
    if code == base + 1:
        return _read_uint(buf, pos, 2)
    if code == base + 2:
        return _read_uint(buf, pos, 4)
    raise ValueError(f"Unsupported msgpack length code: 0x{code:02x}")


def _read_raw(buf: Any, pos: int) -> tuple[bytes, int]:
    code = _byte(buf, pos)
    pos += 1
    if code & 0xE0 == 0xA0:
        length = code & 0x1F
    elif code in (0xC4, 0xC5, 0xC6):
        length, pos = _read_len(buf, pos, code, base=0xC4)
    elif code in (0xD9, 0xDA, 0xDB):
        length, pos = _read_len(buf, pos, code, base=0xD9)
    else:
        raise ValueError(f"Expected msgpack raw/string at offset {pos - 1}, got 0x{code:02x}")
    end = pos + length
    return bytes(buf[pos:end]), end


def _read_array_len(buf: Any, pos: int) -> tuple[int, int]:
    code = _byte(buf, pos)
    pos += 1
    if code & 0xF0 == 0x90:
        return code & 0x0F, pos
    if code == 0xDC:
        return _read_uint(buf, pos, 2)
    if code == 0xDD:
        return _read_uint(buf, pos, 4)
    raise ValueError(f"Expected msgpack array at offset {pos - 1}, got 0x{code:02x}")


def _read_int(buf: Any, pos: int) -> tuple[int, int]:
    code = _byte(buf, pos)
    pos += 1
    if code <= 0x7F:
        return code, pos
    if code >= 0xE0:
        return code - 256, pos
    if code == 0xCC:
        return _read_uint(buf, pos, 1)
    if code == 0xCD:
        return _read_uint(buf, pos, 2)
    if code == 0xCE:
        return _read_uint(buf, pos, 4)
    if code == 0xCF:
        return _read_uint(buf, pos, 8)
    if code == 0xD0:
        value, pos = _read_uint(buf, pos, 1)
        return int.from_bytes(value.to_bytes(1, "big"), "big", signed=True), pos
    if code == 0xD1:
        value, pos = _read_uint(buf, pos, 2)
        return int.from_bytes(value.to_bytes(2, "big"), "big", signed=True), pos
    if code == 0xD2:
        value, pos = _read_uint(buf, pos, 4)
        return int.from_bytes(value.to_bytes(4, "big"), "big", signed=True), pos
    if code == 0xD3:
        value, pos = _read_uint(buf, pos, 8)
        return int.from_bytes(value.to_bytes(8, "big"), "big", signed=True), pos
    raise ValueError(f"Expected msgpack int at offset {pos - 1}, got 0x{code:02x}")


def _read_int_array(buf: Any, pos: int) -> tuple[list[int], int]:
    length, pos = _read_array_len(buf, pos)
    values: list[int] = []
    for _ in range(length):
        value, pos = _read_int(buf, pos)
        values.append(int(value))
    return values, pos


def _read_map_len(buf: Any, pos: int) -> tuple[int, int]:
    code = _byte(buf, pos)
    pos += 1
    if code & 0xF0 == 0x80:
        return code & 0x0F, pos
    if code == 0xDE:
        return _read_uint(buf, pos, 2)
    if code == 0xDF:
        return _read_uint(buf, pos, 4)
    raise ValueError(f"Expected msgpack map at offset {pos - 1}, got 0x{code:02x}")


def _skip_obj(buf: Any, pos: int) -> int:
    code = _byte(buf, pos)
    pos += 1

    if code <= 0x7F or code >= 0xE0 or code in (0xC0, 0xC2, 0xC3):
        return pos
    if code in (0xCC, 0xD0):
        return pos + 1
    if code in (0xCD, 0xD1):
        return pos + 2
    if code in (0xCE, 0xD2, 0xCA):
        return pos + 4
    if code in (0xCF, 0xD3, 0xCB):
        return pos + 8
    if code & 0xE0 == 0xA0:
        return pos + (code & 0x1F)
    if code in (0xC4, 0xC5, 0xC6):
        length, pos = _read_len(buf, pos, code, base=0xC4)
        return pos + length
    if code in (0xD9, 0xDA, 0xDB):
        length, pos = _read_len(buf, pos, code, base=0xD9)
        return pos + length
    if code & 0xF0 == 0x90 or code in (0xDC, 0xDD):
        if code & 0xF0 == 0x90:
            length = code & 0x0F
        elif code == 0xDC:
            length, pos = _read_uint(buf, pos, 2)
        else:
            length, pos = _read_uint(buf, pos, 4)
        for _ in range(length):
            pos = _skip_obj(buf, pos)
        return pos
    if code & 0xF0 == 0x80 or code in (0xDE, 0xDF):
        if code & 0xF0 == 0x80:
            length = code & 0x0F
        elif code == 0xDE:
            length, pos = _read_uint(buf, pos, 2)
        else:
            length, pos = _read_uint(buf, pos, 4)
        for _ in range(length):
            pos = _skip_obj(buf, pos)
            pos = _skip_obj(buf, pos)
        return pos
    fixext_sizes = {0xD4: 1, 0xD5: 2, 0xD6: 4, 0xD7: 8, 0xD8: 16}
    if code in fixext_sizes:
        return pos + 1 + fixext_sizes[code]
    if code in (0xC7, 0xC8, 0xC9):
        length, pos = _read_len(buf, pos, code, base=0xC7)
        return pos + 1 + length
    raise ValueError(f"Unsupported msgpack code at offset {pos - 1}: 0x{code:02x}")


def _msgpack_top_level_int_array(buf: Any, target_key: bytes) -> list[int]:
    map_len, pos = _read_map_len(buf, 0)
    for _ in range(map_len):
        key, pos = _read_raw(buf, pos)
        if key == target_key:
            values, _ = _read_int_array(buf, pos)
            return values
        pos = _skip_obj(buf, pos)
    raise KeyError(f"Missing top-level msgpack array key {target_key!r}")


def _msgpack_numpy_shape(buf: Any, pos: int) -> tuple[tuple[int, ...], int]:
    map_len, pos = _read_map_len(buf, pos)
    shape: tuple[int, ...] | None = None
    for _ in range(map_len):
        key, pos = _read_raw(buf, pos)
        if key == b"shape":
            values, pos = _read_int_array(buf, pos)
            shape = tuple(values)
        else:
            pos = _skip_obj(buf, pos)
    if shape is None:
        raise KeyError("Missing msgpack-numpy `shape` metadata.")
    return shape, pos


def _msgpack_top_level_numpy_shapes(buf: Any, target_keys: tuple[bytes, ...]) -> dict[bytes, tuple[int, ...]]:
    remaining = set(target_keys)
    shapes: dict[bytes, tuple[int, ...]] = {}
    map_len, pos = _read_map_len(buf, 0)
    for _ in range(map_len):
        key, pos = _read_raw(buf, pos)
        if key in remaining:
            shape, pos = _msgpack_numpy_shape(buf, pos)
            shapes[key] = shape
            remaining.remove(key)
            if not remaining:
                return shapes
            continue
        pos = _skip_obj(buf, pos)
    raise KeyError(f"Missing top-level msgpack-numpy key(s) {sorted(remaining)!r}")
