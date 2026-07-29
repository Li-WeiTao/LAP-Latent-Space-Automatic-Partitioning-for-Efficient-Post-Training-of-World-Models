from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

import latent_cluster_train_predictors as trainer  # noqa: E402
from latent_cluster_train_predictors import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    commit_staged_cluster_merge,
    validate_manifest_compatibility,
)
from tworoom_success_rate_eval import (  # noqa: E402
    sha256_file,
    validate_predictor_manifest_artifact,
)


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _fixture(root: Path) -> tuple[Path, Path, Path, dict, dict]:
    artifact_dir = root / "artifact"
    predictor_dir = root / "predictors"
    base = _write(root / "base.ckpt", b"base-checkpoint")
    artifact_files = [
        _write(artifact_dir / "centroids.npy", b"centroids"),
        _write(artifact_dir / "cluster_labels.npz", b"labels"),
        _write(artifact_dir / "cluster_meta.json", b"{}"),
    ]
    artifact = {
        "meta": {
            "assignment_schema_version": 2,
            "transition_label_offset_steps": 0,
            "num_clusters": 2,
        },
        "zscore": None,
    }
    clusters = {}
    for k in range(2):
        ckpt = _write(
            predictor_dir / f"P_train_cluster{k}_object.ckpt",
            f"cluster-{k}".encode(),
        )
        clusters[f"cluster{k}"] = {
            "status": "trained",
            "output_ckpt": str(ckpt.resolve()),
            "output_ckpt_sha256": sha256_file(ckpt),
        }
    manifest = {
        "manifest_schema_version": 2,
        "assignment_schema_version": 2,
        "route_label_offset_steps": 0,
        "cluster_artifact_dir": str(artifact_dir.resolve()),
        "kmeanspp_label_npz": None,
        "cluster_artifact_sha256": {
            str(path.resolve()): sha256_file(path) for path in artifact_files
        },
        "base_checkpoint": {
            "path": str(base.resolve()),
            "sha256": sha256_file(base),
        },
        "clusters": clusters,
    }
    (predictor_dir / "manifest.json").write_text(json.dumps(manifest))
    return artifact_dir, predictor_dir, base, artifact, manifest


def _validate(
    artifact_dir: Path, predictor_dir: Path, base: Path, artifact: dict
) -> Path | None:
    return validate_predictor_manifest_artifact(
        predictor_dir,
        artifact=artifact,
        cluster_artifact_dir=artifact_dir,
        kmeanspp_label_npz=None,
        base_checkpoint=base,
    )


def test_strict_manifest_accepts_exact_bound_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir, predictor_dir, base, artifact, _ = _fixture(Path(tmp))
        assert _validate(artifact_dir, predictor_dir, base, artifact) == (
            predictor_dir / "manifest.json"
        )


def test_strict_manifest_rejects_missing_source_and_hashes() -> None:
    for field in ("cluster_artifact_dir", "cluster_artifact_sha256"):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir, predictor_dir, base, artifact, manifest = _fixture(Path(tmp))
            manifest[field] = None if field == "cluster_artifact_dir" else {}
            (predictor_dir / "manifest.json").write_text(json.dumps(manifest))
            try:
                _validate(artifact_dir, predictor_dir, base, artifact)
            except ValueError:
                pass
            else:
                raise AssertionError(f"missing {field} was accepted")


def test_strict_manifest_cannot_validate_one_checkpoint_and_load_another() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir, predictor_dir, base, artifact, manifest = _fixture(Path(tmp))
        other = _write(Path(tmp) / "other.ckpt", b"other")
        manifest["clusters"]["cluster0"].update(
            output_ckpt=str(other.resolve()),
            output_ckpt_sha256=sha256_file(other),
        )
        (predictor_dir / "manifest.json").write_text(json.dumps(manifest))
        try:
            _validate(artifact_dir, predictor_dir, base, artifact)
        except ValueError as exc:
            assert "selected for loading" in str(exc)
        else:
            raise AssertionError("mismatched loaded checkpoint was accepted")


def test_recovery_fingerprint_rejects_changed_immutable_input() -> None:
    expected = {
        "immutable_config": {"artifact": "A", "seed": 42},
    }
    expected["immutable_config_sha256"] = canonical_json_sha256(
        expected["immutable_config"]
    )
    existing = {
        "manifest_schema_version": 2,
        "immutable_config": {"artifact": "B", "seed": 42},
    }
    existing["immutable_config_sha256"] = canonical_json_sha256(
        existing["immutable_config"]
    )
    try:
        validate_manifest_compatibility(
            existing, expected, manifest_path=Path("manifest.json")
        )
    except RuntimeError as exc:
        assert "artifact" in str(exc)
    else:
        raise AssertionError("incompatible recovery manifest was accepted")


def test_checkpoint_promotion_keyboard_interrupt_restores_previous_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        predictor_dir = root / "predictors"
        staging_dir = predictor_dir / ".staging" / "job"
        staging_dir.mkdir(parents=True)
        final_ckpt = _write(predictor_dir / "P_train_cluster0_object.ckpt", b"old")
        final_meta = _write(predictor_dir / "P_train_cluster0_object.json", b"old-meta")
        staged_ckpt = _write(staging_dir / final_ckpt.name, b"new")
        staged_meta = _write(staging_dir / final_meta.name, b"new-meta")
        immutable = {"artifact": "same", "seed": 42}
        base_manifest = {
            "manifest_schema_version": 2,
            "immutable_config": immutable,
            "immutable_config_sha256": canonical_json_sha256(immutable),
            "clusters": {},
        }
        manifest_path = predictor_dir / "manifest.json"
        atomic_write_json(
            manifest_path,
            {
                **base_manifest,
                "clusters": {
                    "cluster0": {
                        "status": "trained",
                        "output_ckpt": str(final_ckpt.resolve()),
                        "output_ckpt_sha256": sha256_file(final_ckpt),
                        "output_metadata": str(final_meta.resolve()),
                        "output_metadata_sha256": sha256_file(final_meta),
                    }
                },
            },
        )
        original_manifest = manifest_path.read_bytes()
        original_replace = trainer.os.replace
        calls = 0

        def interrupt_second_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("fault injection")
            return original_replace(source, destination)

        trainer.os.replace = interrupt_second_replace
        try:
            try:
                commit_staged_cluster_merge(
                    manifest_path,
                    base_manifest,
                    {"cluster0": {"status": "trained"}},
                    {
                        "cluster0": {
                            "staged_ckpt": staged_ckpt,
                            "staged_metadata": staged_meta,
                            "final_ckpt": final_ckpt,
                            "final_metadata": final_meta,
                        }
                    },
                    partial_recovery=False,
                    allow_full_overwrite=True,
                    staging_dir=staging_dir,
                )
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("fault injection did not interrupt promotion")
        finally:
            trainer.os.replace = original_replace

        assert final_ckpt.read_bytes() == b"old"
        assert final_meta.read_bytes() == b"old-meta"
        assert manifest_path.read_bytes() == original_manifest


def test_directory_fsync_failure_does_not_invalidate_visible_manifest() -> None:
    if trainer.os.name == "nt":
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        original_fsync = trainer.os.fsync
        calls = 0

        def fail_directory_fsync(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("directory fsync unsupported")
            return original_fsync(fd)

        trainer.os.fsync = fail_directory_fsync
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                atomic_write_json(path, {"committed": True})
            assert any("Directory fsync unavailable" in str(w.message) for w in caught)
        finally:
            trainer.os.fsync = original_fsync
        assert json.loads(path.read_text()) == {"committed": True}


if __name__ == "__main__":
    test_strict_manifest_accepts_exact_bound_files()
    test_strict_manifest_rejects_missing_source_and_hashes()
    test_strict_manifest_cannot_validate_one_checkpoint_and_load_another()
    test_recovery_fingerprint_rejects_changed_immutable_input()
    test_checkpoint_promotion_keyboard_interrupt_restores_previous_files()
    test_directory_fsync_failure_does_not_invalidate_visible_manifest()
    print("6 spectral manifest contract tests passed")
