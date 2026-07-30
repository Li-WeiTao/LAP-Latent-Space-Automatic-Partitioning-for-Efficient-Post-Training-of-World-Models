from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lap.encoding.cli import main
from lap.encoding.fast import build_unique_frame_index
from lap.interfaces.encoding import EncodingSelection


class FastEncodingTest(unittest.TestCase):
    def test_exact_index_preserves_legacy_batch_shape_keys(self):
        selection = EncodingSelection(
            sample_ids=np.arange(3),
            frame_ids=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            source_count=3,
        )
        frames, shapes, inverse, unique_count = build_unique_frame_index(
            selection, transition_batch_size=2, exact_batch_shapes=True
        )
        self.assertEqual(unique_count, 4)
        self.assertEqual(len(frames), 5)
        self.assertEqual(set(shapes.tolist()), {2, 4})
        reconstructed = frames[inverse].reshape(selection.frame_ids.shape)
        np.testing.assert_array_equal(reconstructed, selection.frame_ids)

    def test_cli_doctor_and_fixed_batch_encoding(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            dataset_config = tmp_path / "dataset.json"
            encoder_config = tmp_path / "encoder.json"
            dataset_config.write_text('{"offset": 10}', encoding="utf-8")
            encoder_config.write_text('{"scale": 2.0}', encoding="utf-8")
            common = [
                "--dataset-factory",
                "tests.fixtures_encoding:make_dataset",
                "--dataset-config",
                str(dataset_config),
                "--dataset-arg",
                "offset=20",
                "--encoder-factory",
                "tests.fixtures_encoding:make_encoder",
                "--encoder-config",
                str(encoder_config),
                "--encoder-arg",
                "scale=3.0",
                "--pretrained-model",
                "fake-checkpoint",
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--json", "doctor", *common]), 0)
            doctor = json.loads(stdout.getvalue())
            self.assertTrue(doctor["ok"])
            self.assertTrue(doctor["dataset"]["contract_ok"])
            self.assertTrue(doctor["encoder"]["contract_ok"])

            output = tmp_path / "cache.npz"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "--json",
                            "encode",
                            *common,
                            "--output",
                            str(output),
                            "--device",
                            "cpu",
                            "--batch-shape-mode",
                            "fixed",
                            "--frame-batch-size",
                            "3",
                            "--num-workers",
                            "0",
                            "--cpu-threads",
                            "1",
                        ]
                    ),
                    0,
                )
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["ok"])
            self.assertEqual(result["samples"], 3)
            with np.load(output) as data:
                np.testing.assert_array_equal(data["sample_ids"], [20, 21, 22])
                self.assertEqual(data["emb"].shape, (3, 2, 2))
                np.testing.assert_array_equal(
                    data["emb"][:, :, 0], [[0, 3], [3, 6], [6, 9]]
                )
            report = json.loads(
                output.with_suffix(".npz.report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["counts"]["frame_slots"], 6)
            self.assertEqual(report["counts"]["unique_frames"], 4)
            self.assertEqual(
                report["efficiency"]["ideal_unique_frame_speedup"], 1.5
            )


if __name__ == "__main__":
    unittest.main()
