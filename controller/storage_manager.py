# -*- coding: utf-8 -*-
"""
LocalAppStorageManager - output file storage manager anchored to script directory.

Directory structure (auto-created beside the running script/exe):
    <script_dir>/
        picture/    <- screenshots, images
        download/   <- text, files, data exports
"""
import datetime
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalAppStorageManager:
    """Storage manager anchored to the running script/exe directory.

    Path resolution:
        - PyInstaller (sys.frozen): sys.executable directory
        - Normal Python:            __file__ directory (this module)
    """

    PICTURE_DIR = "picture"
    DOWNLOAD_DIR = "download"

    def __init__(self):
        # Anchor to script/exe directory, never os.getcwd()
        if getattr(sys, "frozen", False):
            # PyInstaller bundle
            self._base = Path(sys.executable).resolve().parent
        else:
            # Normal Python: anchor to this file's directory
            self._base = Path(__file__).resolve().parent.parent

        self._picture_dir = self._base / self.PICTURE_DIR
        self._download_dir = self._base / self.DOWNLOAD_DIR
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Silently create output directories if missing."""
        for d in (self._picture_dir, self._download_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.error(
                    "[Storage] Permission denied creating %s - "
                    "please check folder write permissions.", d
                )
            except OSError as e:
                logger.error("[Storage] OS error creating %s: %s", d, e)

    # -- properties --------------------------------------------------------

    @property
    def base_dir(self):
        return self._base

    @property
    def picture_dir(self):
        return self._picture_dir

    @property
    def download_dir(self):
        return self._download_dir

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _timestamp_name(ext="png"):
        """Generate unique filename: YYYYMMDD_HHMMSS_mmm.ext"""
        now = datetime.datetime.now()
        return f"{now.strftime('%Y%m%d_%H%M%S_')}{now.microsecond // 1000:03d}.{ext}"

    # -- public API --------------------------------------------------------

    def save_picture(self, image_data, ext="png"):
        """Save an image to the picture directory.

        Args:
            image_data: PIL Image, or raw bytes (PNG/JPEG/BMP).
            ext: File extension (default 'png').

        Returns:
            Path: absolute path to saved file, or None on failure.
        """
        ext = ext.lstrip(".")
        dest = self._picture_dir / self._timestamp_name(ext)
        try:
            if hasattr(image_data, "save"):
                image_data.save(str(dest))
            elif isinstance(image_data, (bytes, bytearray)):
                dest.write_bytes(image_data)
            else:
                logger.error("[Storage] Unsupported image type: %s", type(image_data))
                return None
            logger.info("[Storage] Picture saved: %s", dest)
            return dest.resolve()
        except PermissionError:
            logger.error(
                "[Storage] Permission denied writing %s - "
                "please check folder write permissions.", dest
            )
        except OSError as e:
            logger.error("[Storage] OS error writing %s: %s", dest, e)
        return None

    def save_download(self, file_content, ext="txt"):
        """Save arbitrary data to the download directory.

        Args:
            file_content: str or bytes content.
            ext: file extension without dot (e.g. 'txt', 'json', 'csv').

        Returns:
            Path: absolute path to saved file, or None on failure.
        """
        ext = ext.lstrip(".")
        dest = self._download_dir / self._timestamp_name(ext)
        try:
            if isinstance(file_content, str):
                dest.write_text(file_content, encoding="utf-8")
            elif isinstance(file_content, (bytes, bytearray)):
                dest.write_bytes(file_content)
            else:
                dest.write_text(str(file_content), encoding="utf-8")
            logger.info("[Storage] Download saved: %s", dest)
            return dest.resolve()
        except PermissionError:
            logger.error(
                "[Storage] Permission denied writing %s - "
                "please check folder write permissions.", dest
            )
        except OSError as e:
            logger.error("[Storage] OS error writing %s: %s", dest, e)
        return None

    def list_recent(self, category="picture", n=10):
        """List most recent files in a category directory."""
        target = self._picture_dir if category == "picture" else self._download_dir
        if not target.exists():
            return []
        files = sorted(target.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:n]

    def get_usage(self):
        """Return disk usage summary dict."""
        result = {"total_files": 0, "total_bytes": 0, "picture_count": 0, "download_count": 0}
        if self._picture_dir.exists():
            for f in self._picture_dir.iterdir():
                if f.is_file():
                    result["picture_count"] += 1
                    result["total_files"] += 1
                    result["total_bytes"] += f.stat().st_size
        if self._download_dir.exists():
            for f in self._download_dir.iterdir():
                if f.is_file():
                    result["download_count"] += 1
                    result["total_files"] += 1
                    result["total_bytes"] += f.stat().st_size
        return result


# backward-compatible alias
StorageManager = LocalAppStorageManager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sm = LocalAppStorageManager()

    print(f"Base dir:     {sm.base_dir}")
    print(f"Picture dir:  {sm.picture_dir}")
    print(f"Download dir: {sm.download_dir}")

    # test picture save (bytes)
    p = sm.save_picture(b"\x89PNG mock image data", ext="png")
    print(f"Picture test: {p}")

    # test download save (text)
    p = sm.save_download("Hello from LocalAppStorageManager!", ext="txt")
    print(f"Download test: {p}")

    # test PIL if available
    try:
        from PIL import Image
        img = Image.new("RGB", (200, 100), color="cyan")
        p = sm.save_picture(img)
        print(f"PIL test: {p}")
    except ImportError:
        print("PIL not available, skipped")

    print("Recent downloads:", sm.list_recent("download", 3))
    print("Usage:", sm.get_usage())
