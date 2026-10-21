"""Seekable HTTP file for zipfile: every read is an HTTP Range request.

Usage:
    z = zipfile.ZipFile(RemoteFile(url))
    data = z.read("COLOR_BODY/0030_....avi")

Reads only the zip directory and the entries you ask for. The URL is requested
fresh on every read, so signed redirects that expire (Figshare, ~10 s) still work.
"""
import io
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}  # Mendeley returns 403 without it


def _get(url, start, end, tries=6):
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=60)
        except OSError:  # URLError, timeouts, SSL handshake
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


class RemoteFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        with _get(url, 0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        with _get(self.url, self.pos, end) as r:
            data = r.read()
        self.pos += len(data)
        return data


if __name__ == "__main__":
    # List a remote zip: python remote_zip.py <url> [n_entries]
    import sys
    import zipfile

    z = zipfile.ZipFile(RemoteFile(sys.argv[1]))
    infos = z.infolist()
    print("entries", len(infos))
    for i in infos[: int(sys.argv[2]) if len(sys.argv) > 2 else 20]:
        print(i.filename, i.file_size)
