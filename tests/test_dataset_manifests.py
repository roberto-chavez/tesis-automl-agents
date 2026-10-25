import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from automl_agents.dataset import DatasetError, prepare_dataset, profile_dataset, split_files
from automl_agents.demo import make_demo_dataset
from automl_agents.schemas import DatasetManifest, SplitManifest


def prepare(ds, run, **kwargs):
    return prepare_dataset(ds, run, kwargs.get("val", .15), kwargs.get("test", .15), kwargs.get("seed", 42))


def assert_code(exc, code):
    assert exc.value.code == code
    assert exc.value.to_dict()["recoverable"] is False


def test_all_images_are_decoded_not_only_sampled(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    (ds / "circulo" / "corrupt.png").write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="corrupt.png") as exc:
        profile_dataset(ds, sample=1)
    assert_code(exc, "corrupt_image")
    assert not (tmp_path / "run" / "data").exists()


def test_truncated_image_rejected(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    p = ds / "circulo" / "000.png"
    p.write_bytes(p.read_bytes()[:60])
    with pytest.raises(DatasetError) as exc:
        prepare(ds, tmp_path / "run")
    assert_code(exc, "corrupt_image")


@pytest.mark.parametrize("n,val,test", [(6, .15, .15), (10, .01, .15), (10, .15, .01)])
def test_small_class_fails_for_requested_fractions(tmp_path, n, val, test):
    ds = make_demo_dataset(tmp_path / "ds", per_class=n)
    with pytest.raises(DatasetError, match="imágenes únicas") as exc:
        prepare(ds, tmp_path / "run", val=val, test=test)
    assert_code(exc, "insufficient_class_images")
    assert not (tmp_path / "run" / "data").exists()


def test_three_unique_images_support_thirds(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=3)
    _, _, splits = prepare(ds, tmp_path / "run", val=1/3, test=1/3)
    assert all(len(paths) == 3 for paths, _ in splits.values())


def test_duplicates_never_cross_splits(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    for i in range(15):
        shutil.copyfile(ds / "circulo" / "000.png", ds / "circulo" / f"copy{i}.png")
    _, files, splits = prepare(ds, tmp_path / "run")
    hashes = []
    import hashlib
    for paths, _ in splits.values():
        hashes.append({hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    assert all(hashes[i].isdisjoint(hashes[j]) for i in range(3) for j in range(i))
    assert sum(len(paths) for paths, _ in splits.values()) == 45
    # La API previa conserva la misma protección.
    assert split_files(files, list(files), .15, .15, 42) == splits


def test_duplicates_do_not_inflate_viable_class_size(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=3)
    for i in range(20):
        shutil.copyfile(ds / "circulo" / "000.png", ds / "circulo" / f"copy{i}.png")
    with pytest.raises(DatasetError) as exc:
        prepare(ds, tmp_path / "run")
    assert_code(exc, "insufficient_class_images")


def test_duplicate_label_conflict(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    shutil.copyfile(ds / "circulo" / "000.png", ds / "cuadrado" / "copy.png")
    with pytest.raises(DatasetError, match="conflicto") as exc:
        prepare(ds, tmp_path / "run")
    assert_code(exc, "duplicate_label_conflict")


def test_resume_reads_manifests_without_resplitting(tmp_path, monkeypatch):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    run = tmp_path / "run"
    profile, files, splits = prepare(ds, run)
    manifest_path = run / "data" / "dataset_manifest.json"
    partition_path = run / "data" / "split_manifest.json"
    saved = manifest_path.read_bytes(), partition_path.read_bytes()
    manifest = DatasetManifest.model_validate_json(saved[0])
    partition = SplitManifest.model_validate_json(saved[1])
    assert all(not Path(e.path).is_absolute() and e.sha256 for e in manifest.images)
    assert partition.seed == 42
    def fail(*args, **kwargs):
        pytest.fail("No debe recalcularse la partición al reanudar")
    monkeypatch.setattr("automl_agents.dataset._partition", fail)
    assert prepare(ds, run) == (profile, files, splits)
    assert saved == (manifest_path.read_bytes(), partition_path.read_bytes())
    # Las referencias relativas permiten mover el dataset intacto.
    moved = tmp_path / "moved"
    shutil.copytree(ds, moved)
    _, _, relocated = prepare(moved, run)
    for name in splits:
        assert [Path(p).relative_to(ds) for p in splits[name][0]] == [
            Path(p).relative_to(moved) for p in relocated[name][0]]


@pytest.mark.parametrize("mutation", ["edit", "add", "remove", "rename_class", "empty_class"])
def test_resume_rejects_dataset_changes(tmp_path, mutation):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    run = tmp_path / "run"
    prepare(ds, run)
    p = ds / "circulo" / "000.png"
    if mutation == "edit":
        Image.new("RGB", (51, 51), "red").save(p)
    elif mutation == "add":
        shutil.copyfile(p, p.with_name("extra.png"))
    elif mutation == "remove":
        p.unlink()
    elif mutation == "rename_class":
        (ds / "circulo").rename(ds / "other")
    else:
        (ds / "empty").mkdir()
    with pytest.raises(DatasetError) as exc:
        prepare(ds, run)
    assert exc.value.code in {"dataset_changed", "empty_class"}


@pytest.mark.parametrize("kwargs", [{"seed": 43}, {"val": .2}, {"test": .2}])
def test_resume_rejects_config_changes(tmp_path, kwargs):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    prepare(ds, tmp_path / "run")
    with pytest.raises(DatasetError) as exc:
        prepare(ds, tmp_path / "run", **kwargs)
    assert_code(exc, "split_config_changed")


@pytest.mark.parametrize("damage", ["missing", "bad_json", "fingerprint", "duplicate", "coverage", "escape", "leak"])
def test_invalid_or_interrupted_manifests_fail_closed(tmp_path, damage):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    shutil.copyfile(ds / "circulo" / "000.png", ds / "circulo" / "copy.png")
    run = tmp_path / "run"
    prepare(ds, run)
    path = run / "data" / "split_manifest.json"
    data = json.loads(path.read_text("utf-8"))
    if damage == "missing":
        path.unlink()
    elif damage == "bad_json":
        path.write_text("{", encoding="utf-8")
    else:
        if damage == "fingerprint":
            data["dataset_sha256"] = "0" * 64
        elif damage == "duplicate":
            data["train"].append(data["val"][0])
        elif damage == "coverage":
            data["train"].pop()
        elif damage == "escape":
            data["train"][0] = "../../outside.png"
        else:
            source = next(k for k in ("train", "val", "test") if "circulo/copy.png" in data[k])
            dest = "val" if source == "train" else "train"
            data[source].remove("circulo/copy.png")
            data[dest].append("circulo/copy.png")
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError) as exc:
        prepare(ds, run)
    assert_code(exc, "invalid_manifest")


def test_agent_profile_uses_only_training(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10, size=48)
    run = tmp_path / "run"
    _, _, splits = prepare(ds, run)
    # Nueva ejecución con las mismas asignaciones: solo held-out contiene tamaños pequeños.
    shutil.rmtree(run)
    for split_index, split in enumerate(("val", "test")):
        for index, filename in enumerate(splits[split][0]):
            Image.new("RGB", (8 + index, 9 + index), (index, split_index, 0)).save(filename)
    profile, _, final = prepare(ds, run)
    assert final == splits
    assert profile.scope == "train"
    assert profile.n_images == len(splits["train"][0])
    assert profile.median_size == (48, 48) and profile.min_side == 48
    assert profile.counts == {c: 8 for c in profile.classes}
    profile.n_images = 1200
    assert profile.size_bucket(.1) == "medium"


def test_readable_but_unsupported_format_rejected(tmp_path):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    Image.new("RGB", (40, 40)).save(ds / "circulo" / "fake.png", format="TIFF")
    with pytest.raises(DatasetError, match="no compatible") as exc:
        prepare(ds, tmp_path / "run")
    assert_code(exc, "corrupt_image")


@pytest.mark.parametrize("val,test", [(0, .15), (.6, .4), (-.1, .2), (float("nan"), .15)])
def test_invalid_fractions_are_friendly_errors(tmp_path, val, test):
    ds = make_demo_dataset(tmp_path / "ds", per_class=10)
    with pytest.raises(DatasetError) as exc:
        prepare(ds, tmp_path / "run", val=val, test=test)
    assert_code(exc, "invalid_split_fractions")
