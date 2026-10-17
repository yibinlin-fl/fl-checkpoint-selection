"""OpenI transport contract: verified inputs, safe extraction, recover-before-analysis."""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import zipfile

from .artifacts import sha256


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run_id")
    return run_id


def checked_context(prepare, upload_output):
    if not callable(upload_output):
        raise RuntimeError("c2net upload_output unavailable; refusing training")
    context = prepare()
    if context is None or not getattr(context, "output_path", None):
        raise RuntimeError("c2net output context unavailable; refusing training")
    return context


def verify_file(path, expected_bytes, expected_sha256):
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Invalid SHA256")
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
        raise ValueError("Input archive bytes/SHA256 mismatch")
    return path


def safe_extract_zip(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Extraction needs a new directory")
    with zipfile.ZipFile(archive) as z:
        names = set()
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (name.is_absolute() or ".." in name.parts or "\\" in item.filename or ":" in item.filename
                    or stat.S_ISLNK(mode) or item.filename in names):
                raise ValueError("Unsafe or duplicate archive member")
            names.add(item.filename)
        destination.mkdir(parents=True)
        for item in z.infolist():
            path = destination.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as src, path.open("xb") as out:
                    shutil.copyfileobj(src, out)
    return destination


def verify_input_tree(root):
    root = Path(root)
    manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
    listed = set()
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe manifest path")
        path = root.joinpath(*relative.parts)
        if path.is_symlink():
            raise ValueError("Symlink input")
        verify_file(path, expected["bytes"], expected["sha256"])
        listed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != listed | {"input_manifest.json"}:
        raise ValueError("Input package contains unlisted or missing files")
    return manifest


def package_results(run_root, archive):
    run_root, archive = Path(run_root), Path(archive)
    if archive.exists():
        raise FileExistsError("Refusing result archive overwrite")
    files = sorted(p for p in run_root.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in files):
        raise ValueError("Result symlinks forbidden")
    manifest_path = run_root / "metadata" / "artifact_manifest.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps({p.relative_to(run_root).as_posix(): {
        "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files if p != manifest_path}, indent=2), encoding="utf-8")
    files = sorted(p for p in run_root.rglob("*") if p.is_file())
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(run_root).as_posix(), recursive=False)
    return archive


def publish(paths, output_path, upload_output, *, mutable_status=None):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    for source in map(Path, paths):
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / source.name
        if destination.resolve() != source.resolve():
            if destination.exists() and source.name != mutable_status:
                raise FileExistsError("Remote staging filename already exists")
            shutil.copy2(source, destination)
    result = upload_output()
    if result is False or (isinstance(result, dict) and result.get("success") is False):
        raise RuntimeError("c2net reported upload failure")
    # A normal SDK return is recorded, never claimed to prove remote download integrity.
    return {"upload_call_returned": True, "remote_download_verified": False}
