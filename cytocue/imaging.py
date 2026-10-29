from __future__ import annotations
from collections import OrderedDict
import numpy as np
import tifffile


class TiffReader:

    def __init__(self, path, cache_bytes=64 * 1024**2):
        self.path = str(path)
        self.file = tifffile.TiffFile(path)
        self.page = self.file.pages[0]
        p = self.page
        if p.samplesperpixel != 1 or len(p.shape) != 2:
            raise ValueError(f"Expected scalar 2D TIFF: {path}, {p.shape}")
        self.shape, self.dtype = (tuple(p.shape), np.dtype(p.dtype))
        self.tile_h = p.tilelength if p.is_tiled else p.rowsperstrip
        self.tile_w = p.tilewidth if p.is_tiled else self.shape[1]
        self.cols = (self.shape[1] + self.tile_w - 1) // self.tile_w
        self.cache = OrderedDict()
        self.cache_bytes = cache_bytes
        self.cached_bytes = 0
        self.memory = None
        if int(p.compression) == 1:
            try:
                self.memory = tifffile.memmap(path, page=0, mode="r")
            except ValueError:
                pass

    def segment(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        p, handle = (self.page, self.file.filehandle)
        handle.seek(p.dataoffsets[index])
        raw = handle.read(p.databytecounts[index])
        array, _, _ = p.decode(raw, index, jpegtables=p.jpegtables)
        if array is None:
            array = np.zeros((self.tile_h, self.tile_w), dtype=self.dtype)
        while array.ndim > 2:
            if array.shape[0] == 1:
                array = array[0]
            elif array.shape[-1] == 1:
                array = array[..., 0]
            else:
                raise ValueError(f"Non-scalar TIFF segment {array.shape}")
        self.cache[index] = array
        self.cached_bytes += array.nbytes
        while len(self.cache) > 1 and self.cached_bytes > self.cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.cached_bytes -= old.nbytes
        return array

    def read(self, y0, x0, y1, x1):
        y0, x0, y1, x1 = map(int, (y0, x0, y1, x1))
        out = np.zeros((max(0, y1 - y0), max(0, x1 - x0)), dtype=self.dtype)
        ya, xa = (max(0, y0), max(0, x0))
        yb, xb = (min(self.shape[0], y1), min(self.shape[1], x1))
        if ya >= yb or xa >= xb:
            return out
        if self.memory is not None:
            out[ya - y0 : yb - y0, xa - x0 : xb - x0] = self.memory[ya:yb, xa:xb]
            return out
        for ty in range(ya // self.tile_h, (yb - 1) // self.tile_h + 1):
            for tx in range(xa // self.tile_w, (xb - 1) // self.tile_w + 1):
                by, bx = (ty * self.tile_h, tx * self.tile_w)
                segment = self.segment(ty * self.cols + tx)
                top, left = (max(ya, by), max(xa, bx))
                bottom, right = (min(yb, by + segment.shape[0]), min(xb, bx + segment.shape[1]))
                out[top - y0 : bottom - y0, left - x0 : right - x0] = segment[
                    top - by : bottom - by, left - bx : right - bx
                ]
        return out

    def blocks(self, rows=256):
        for y in range(0, self.shape[0], rows):
            yield (y, self.read(y, 0, min(y + rows, self.shape[0]), self.shape[1]))

    def coverage(self, y0, x0, y1, x1):
        yy, xx = np.ogrid[y0:y1, x0:x1]
        return (yy >= 0) & (yy < self.shape[0]) & (xx >= 0) & (xx < self.shape[1])

    def close(self):
        if self.memory is not None:
            self.memory._mmap.close()
            self.memory = None
        self.file.close()
        self.cache.clear()


def mask_cells(reader):
    cells = {}
    for origin, block in reader.blocks(rows=64):
        if (
            not np.isfinite(block).all()
            or (block < 0).any()
            or (not np.equal(block, np.floor(block)).all())
        ):
            raise ValueError("Mask must contain finite nonnegative integer IDs")
        yy, xx = np.nonzero(block)
        if not len(yy):
            continue
        vals = block[yy, xx].astype(np.int64)
        ids, inverse, counts = np.unique(vals, return_inverse=True, return_counts=True)
        ymin = np.full(len(ids), block.shape[0], np.int64)
        xmin = np.full(len(ids), block.shape[1], np.int64)
        ymax = np.zeros(len(ids), np.int64)
        xmax = np.zeros(len(ids), np.int64)
        np.minimum.at(ymin, inverse, yy)
        np.minimum.at(xmin, inverse, xx)
        np.maximum.at(ymax, inverse, yy)
        np.maximum.at(xmax, inverse, xx)
        for i, cid in enumerate(ids):
            cid = int(cid)
            value = [
                int(ymin[i] + origin),
                int(xmin[i]),
                int(ymax[i] + origin + 1),
                int(xmax[i] + 1),
                int(counts[i]),
            ]
            if cid in cells:
                v = cells[cid]
                v[0] = min(v[0], value[0])
                v[1] = min(v[1], value[1])
                v[2] = max(v[2], value[2])
                v[3] = max(v[3], value[3])
                v[4] += value[4]
            else:
                cells[cid] = value
    return cells
