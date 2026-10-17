import json
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import pytest

from p1.artifacts import sha256
from p1.openi import checked_context, package_results, publish, safe_extract_zip, validate_run_id, verify_input_tree


@pytest.mark.parametrize("bad", ["../run", "run/name", "", "a:b"])
def test_run_id_blocks_path_injection(bad):
    with pytest.raises(ValueError):
        validate_run_id(bad)


def test_missing_context_stops_before_training():
    with pytest.raises(RuntimeError, match="context"):
        checked_context(lambda: None, lambda: None)


@pytest.mark.parametrize("name,symlink", [("../escape",False),("/absolute",False),("normal",True)])
def test_zip_rejects_escape_and_symlinks(tmp_path,name,symlink):
    archive=tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as z:
        item=zipfile.ZipInfo(name)
        if symlink:
            item.create_system=3
            item.external_attr=(stat.S_IFLNK|0o777)<<16
        z.writestr(item,"payload")
    with pytest.raises(ValueError):
        safe_extract_zip(archive,tmp_path/"extract")
    assert not (tmp_path/"extract").exists()


def test_input_archive_manifest_and_unknown_files(tmp_path):
    root=tmp_path/"input"; root.mkdir()
    (root/"payload").write_bytes(b"actual input")
    (root/"input_manifest.json").write_text(json.dumps({"files":{"payload":{
        "bytes":12,"sha256":sha256(root/"payload")}}}))
    verify_input_tree(root)
    (root/"extra").write_bytes(b"unlisted")
    with pytest.raises(ValueError,match="unlisted"):
        verify_input_tree(root)


def test_recovery_contains_checkpoint_and_upload_failure_is_not_success(tmp_path):
    run=tmp_path/"run"; (run/"results/checkpoints").mkdir(parents=True)
    (run/"results/checkpoints/round_0002.pt").write_bytes(b"fixture checkpoint")
    archive=package_results(run,tmp_path/"recovery.tar.gz")
    import tarfile
    with tarfile.open(archive) as tar:
        assert "results/checkpoints/round_0002.pt" in tar.getnames()
        assert "metadata/artifact_manifest.json" in tar.getnames()
    with pytest.raises(RuntimeError,match="failure"):
        publish([archive],tmp_path/"output",lambda:False)


@pytest.mark.parametrize("fail_at", [1, None, 2])
def test_openi_recovery_upload_orders_analysis_and_handles_failure(tmp_path, monkeypatch, fail_at):
    import sys, types
    import scripts.openi_phase1a as entry
    context=SimpleNamespace(output_path=str(tmp_path/"published"),dataset_path=str(tmp_path))
    calls=[]
    sdk=types.ModuleType("c2net.context")
    sdk.prepare=lambda:context
    def upload():
        calls.append("upload")
        if calls.count("upload")==fail_at:
            raise RuntimeError("mock transport failure")
    sdk.upload_output=upload
    monkeypatch.setitem(sys.modules,"c2net",types.ModuleType("c2net"))
    monkeypatch.setitem(sys.modules,"c2net.context",sdk)
    monkeypatch.setattr(entry,"require_v100",lambda:{"name":"MOCK ONLY"})
    monkeypatch.setattr(entry,"verify_file",lambda *a:None)
    monkeypatch.setattr(entry,"safe_extract_zip",lambda *a:None)
    monkeypatch.setattr(entry,"verify_input_tree",lambda *a:{"files":{}})
    monkeypatch.setattr(entry,"verify_training",lambda *a:None)
    monkeypatch.setattr(entry,"verify_frozen",lambda *a:None)
    monkeypatch.setattr(entry,"summarize_cost",lambda *a:None)
    import scripts.replay_phase1a_validation
    monkeypatch.setattr(scripts.replay_phase1a_validation,"replay",lambda *a: calls.append("replay"))
    import torchvision.datasets.utils
    monkeypatch.setattr(torchvision.datasets.utils,"check_integrity",lambda *a:True)
    monkeypatch.setattr(torchvision.datasets.utils,"extract_archive",lambda *a,**k:None)
    monkeypatch.setattr(entry.subprocess,"run",lambda *a,**k:None)
    import p1.training
    monkeypatch.setattr(p1.training,"git_revision",lambda:"mock-only")
    class Process:
        stdout=[]
        def wait(self): return 0
    def popen(command,**kw):
        calls.append(command[3])
        return Process()
    monkeypatch.setattr(entry.subprocess,"Popen",popen)
    def pack(root,archive):
        archive.write_bytes(b"mock result archive");return archive
    monkeypatch.setattr(entry,"package_results",pack)
    code=entry.main(["--run_id","mock_recovery_failure","--work_root",str(tmp_path/"work"),
                     "--data_source",str(tmp_path/"input.zip"),"--input_bytes","1","--input_sha256","0"*64])
    assert code == (1 if fail_at else 0)
    assert "train" in calls
    status=json.loads((tmp_path/"work/mock_recovery_failure/p1_mock_recovery_failure_status.json").read_text())
    assert status["scientific_evidence"] is False
    if fail_at == 1:
        assert "evaluate" not in calls
        assert status["execution_status"] == "failed"
    elif fail_at == 2:
        assert "evaluate" in calls
        assert status["analysis_status"] == "complete"
        assert status["execution_status"] == "failed"
        assert status["failed_stage"] == "final_publication"
    else:
        assert calls.index("train") < calls.index("upload") < calls.index("evaluate")
        assert status["analysis_status"] == "complete"
        assert status["execution_status"] == "complete"
