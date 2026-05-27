"""
Tests for tools/model_manager — no GPU or real model weights required.
Run with: pytest tests/test_model_manager.py -v
"""
import multiprocessing as mp
import pytest
import os
import pickle
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_channel():
    from tools.model_manager.server import ProcessChannel
    return ProcessChannel()


# ═════════════════════════════════════════════════════════════════════════════
# auto_batched
# ═════════════════════════════════════════════════════════════════════════════

class TestAutoBatched(unittest.TestCase):
    def setUp(self):
        from tools.model_manager.server import auto_batched
        self.auto_batched = auto_batched

    def test_tuple_args(self):
        result = self.auto_batched([(1, "a"), (2, "b"), (3, "c")])
        self.assertEqual(result, ([1, 2, 3], ["a", "b", "c"]))

    def test_dict_args(self):
        result = self.auto_batched([{"x": 1, "y": True}, {"x": 2, "y": True}])
        self.assertEqual(result, {"x": [1, 2], "y": True})

    def test_dict_bool_is_not_batched(self):
        result = self.auto_batched([{"flag": True}, {"flag": True}])
        self.assertTrue(result["flag"] is True)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.auto_batched([[1, 2], [3, 4]])


# ═════════════════════════════════════════════════════════════════════════════
# ProcessChannel
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessChannelPickle(unittest.TestCase):
    def test_getstate_excludes_manager(self):
        ch = _make_channel()
        state = ch.__getstate__()
        self.assertNotIn("_manager", state)
        ch.shutdown()

    def test_setstate_sets_manager_none(self):
        ch = _make_channel()
        state = ch.__getstate__()
        ch2 = object.__new__(ch.__class__)
        ch2.__setstate__(state)
        self.assertIsNone(ch2._manager)
        ch.shutdown()

    def test_round_trip_pickle(self):
        # mp.Queue / mp.Condition require a spawn context to pickle;
        # verify __getstate__ produces the right shape instead.
        ch = _make_channel()
        state = ch.__getstate__()
        self.assertNotIn("_manager", state)
        self.assertIn("input", state)
        self.assertIn("output", state)
        self.assertIn("cond_c", state)
        self.assertIn("cond_s", state)
        ch.shutdown()


class TestProcessChannelPutGetC(unittest.TestCase):
    """put_c / get_c round-trip in a background thread (simulates client side)."""

    def setUp(self):
        self.ch = _make_channel()

    def tearDown(self):
        self.ch.shutdown()

    def _server_reply(self, expected_url, reply_val):
        """Simulate server: read one item from get_s, put reply via put_s."""
        url, vals = self.ch.get_s()
        # batched urls return a list of (id, (args, kwargs))
        for req_id, (args, kwargs) in vals:
            self.ch.put_s([(req_id, reply_val)])

    def test_ram_round_trip(self):
        reply = ["cat", "dog"]
        t = threading.Thread(target=self._server_reply, args=("/ram", reply))
        t.start()
        req_id = 42
        self.ch.put_c(req_id, "/ram", (("img_data",), {}))
        result = self.ch.get_c(req_id)
        t.join(timeout=3)
        self.assertEqual(result, reply)

    def test_multiple_clients_get_correct_replies(self):
        """Two requests with different IDs should each get their own reply."""
        results = {}

        def server():
            replied = 0
            while replied < 2:
                url, vals = self.ch.get_s()
                for req_id, (args, kwargs) in vals:
                    self.ch.put_s([(req_id, f"reply_{req_id}")])
                    replied += 1

        t = threading.Thread(target=server)
        t.start()

        def client(req_id):
            self.ch.put_c(req_id, "/ram", ((f"img_{req_id}",), {}))
            results[req_id] = self.ch.get_c(req_id)

        threads = [threading.Thread(target=client, args=(i,)) for i in [100, 200]]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=3)
        t.join(timeout=3)

        self.assertEqual(results[100], "reply_100")
        self.assertEqual(results[200], "reply_200")


class TestProcessChannelGetS(unittest.TestCase):
    def setUp(self):
        self.ch = _make_channel()

    def tearDown(self):
        self.ch.shutdown()

    def _get_s_in_thread(self, model_subset=None, timeout=2):
        """Run get_s in a thread and return (url, vals)."""
        result = {}

        def worker():
            result['val'] = self.ch.get_s(model_subset=model_subset)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return result.get('val')

    def test_batches_up_to_four(self):
        # put 6 items via put_c (which notifies cond_s), then read in a thread
        for i in range(6):
            self.ch.put_c(i, "/ram", ((f"img_{i}",), {}))
        result = self._get_s_in_thread()
        self.assertIsNotNone(result)
        url, vals = result
        self.assertEqual(url, "/ram")
        self.assertLessEqual(len(vals), 4)
        self.assertGreaterEqual(len(vals), 1)

    def test_unbatched_sam_returns_single(self):
        self.ch.put_c(99, "/sam", (("img",), {}))
        result = self._get_s_in_thread()
        self.assertIsNotNone(result)
        url, val = result
        self.assertEqual(url, "/sam")
        req_id, _ = val
        self.assertEqual(req_id, 99)

    def test_unbatched_clip_image_returns_single(self):
        self.ch.put_c(77, "/clip/image", (("img",), {}))
        result = self._get_s_in_thread()
        self.assertIsNotNone(result)
        url, val = result
        self.assertEqual(url, "/clip/image")

    def test_model_subset_filters_urls(self):
        # put the item before starting the thread to avoid the race
        # between put_c's notify and get_s acquiring cond_s
        self.ch.put_c(1, "/ram", (("img",), {}))
        time.sleep(0.01)  # let notify fire before thread enters cond_s.wait
        result = self._get_s_in_thread(model_subset=["ram"], timeout=3)
        self.assertIsNotNone(result)
        url, vals = result
        self.assertEqual(url, "/ram")

    def test_model_subset_blocks_irrelevant(self):
        # /completion item should be ignored when subset=["ram"];
        # unblock after a short delay by adding a /ram item
        self.ch.put_c(1, "/completion", (("text",), {}))

        def unblock():
            time.sleep(0.05)
            self.ch.put_c(2, "/ram", (("img",), {}))

        t = threading.Thread(target=unblock, daemon=True)
        t.start()
        result = self._get_s_in_thread(model_subset=["ram"], timeout=3)
        t.join(timeout=1)
        self.assertIsNotNone(result)
        url, vals = result
        self.assertEqual(url, "/ram")


# ═════════════════════════════════════════════════════════════════════════════
# ModelProcess (logic only — no actual model loading)
# ═════════════════════════════════════════════════════════════════════════════

class TestModelProcessDeviceSelection(unittest.TestCase):
    def _make_process(self, cuda_devices, device_override=None):
        from tools.model_manager.server import ModelProcess, ProcessChannel
        ch = MagicMock(spec=ProcessChannel)
        return ModelProcess(ch, cuda_devices=cuda_devices, device=device_override)

    def test_device_cuda_when_cuda_devices_present(self):
        proc = self._make_process(cuda_devices=["0"])
        device = proc._device_override or ("cuda" if proc.cuda_devices else "cpu")
        self.assertEqual(device, "cuda")

    def test_device_cpu_when_no_cuda_no_override(self):
        proc = self._make_process(cuda_devices=[])
        device = proc._device_override or ("cuda" if proc.cuda_devices else "cpu")
        self.assertEqual(device, "cpu")

    def test_device_override_mps(self):
        proc = self._make_process(cuda_devices=[], device_override="mps")
        device = proc._device_override or ("cuda" if proc.cuda_devices else "cpu")
        self.assertEqual(device, "mps")

    def test_cuda_visible_devices_set_for_cuda(self):
        proc = self._make_process(cuda_devices=["1"])
        with patch.dict(os.environ, {}, clear=False):
            proc.init()
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "1")

    def test_cuda_visible_devices_empty_for_cpu_mode(self):
        proc = self._make_process(cuda_devices=[])
        with patch.dict(os.environ, {}, clear=False):
            proc.init()
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")

    def test_cuda_visible_devices_untouched_for_mps(self):
        proc = self._make_process(cuda_devices=[], device_override="mps")
        original = os.environ.get("CUDA_VISIBLE_DEVICES", "__unset__")
        proc.init()
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES", "__unset__"), original)

    def test_load_model_unknown_raises(self):
        proc = self._make_process(cuda_devices=[])
        proc.init()
        with self.assertRaises(ValueError):
            proc._load_model("unknown_model")

    def test_model_subset_skips_excluded(self):
        proc = self._make_process(cuda_devices=[], device_override="cpu")
        proc.model_subset = ["ram", "dino"]
        proc.init()
        result = proc._load_model("clip")
        self.assertIsNone(result)


# ═════════════════════════════════════════════════════════════════════════════
# ModelManager
# ═════════════════════════════════════════════════════════════════════════════

class TestModelManagerRemote(unittest.TestCase):
    """local=False: clients created with channel=None (HTTP mode)."""

    def setUp(self):
        from tools.model_manager import ModelManager
        self.mm = ModelManager(local=False)

    def tearDown(self):
        self.mm.close()

    def test_all_clients_present(self):
        for name in ["ram", "dino", "sam", "clip", "embedding", "completion"]:
            self.assertIn(name, self.mm.models)

    def test_clients_have_no_channel(self):
        for client in self.mm.models.values():
            self.assertIsNone(client.server_channel)

    def test_get_model_returns_client(self):
        from tools.model_manager.client import RAMClient
        self.assertIsInstance(self.mm.get_model("ram"), RAMClient)

    def test_get_model_unknown_raises(self):
        with self.assertRaises(ValueError):
            self.mm.get_model("nonexistent")

    def test_device_stored(self):
        from tools.model_manager import ModelManager
        mm = ModelManager(device="cpu", local=False)
        self.assertEqual(mm.device, "cpu")
        mm.close()


class TestModelManagerSetChannel(unittest.TestCase):
    def setUp(self):
        from tools.model_manager import ModelManager
        self.mm = ModelManager(local=False)

    def tearDown(self):
        self.mm.close()

    def test_set_channel_propagates_to_all_clients(self):
        fake_channel = MagicMock()
        self.mm.set_channel(fake_channel)
        for client in self.mm.models.values():
            self.assertIs(client.server_channel, fake_channel)

    def test_set_channel_updates_manager_channel(self):
        fake_channel = MagicMock()
        self.mm.set_channel(fake_channel)
        self.assertIs(self.mm._channel, fake_channel)

    def test_set_channel_updates_device(self):
        fake_channel = MagicMock()
        self.mm.set_channel(fake_channel, device="mps")
        self.assertEqual(self.mm.device, "mps")
        for client in self.mm.models.values():
            self.assertEqual(client.device, "mps")

    def test_set_channel_none_clears_channel(self):
        fake_channel = MagicMock()
        self.mm.set_channel(fake_channel)
        self.mm.set_channel(None)
        for client in self.mm.models.values():
            self.assertIsNone(client.server_channel)


class TestModelManagerClose(unittest.TestCase):
    def test_close_clears_models(self):
        from tools.model_manager import ModelManager
        mm = ModelManager(local=False)
        mm.close()
        self.assertEqual(mm.models, {})

    def test_close_clears_channel(self):
        from tools.model_manager import ModelManager
        mm = ModelManager(local=False)
        fake_ch = MagicMock()
        fake_ch.shutdown = MagicMock()
        mm._channel = fake_ch
        mm.close()
        self.assertIsNone(mm._channel)
        fake_ch.shutdown.assert_called_once()

    def test_close_terminates_processes(self):
        from tools.model_manager import ModelManager
        mm = ModelManager(local=False)
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mm._processes = [mock_proc]
        mm.close()
        mock_proc.terminate.assert_called_once()


class TestModelManagerLocalInit(unittest.TestCase):
    """local=True: process spawned, clients get channel."""

    @patch("tools.model_manager.server.ModelProcess.start")
    def test_local_init_no_cuda_creates_channel(self, mock_start):
        from tools.model_manager import ModelManager
        with patch("torch.cuda.device_count", return_value=0), \
             patch("torch.backends.mps.is_available", return_value=False):
            mm = ModelManager(device="cpu", local=True)
            self.assertIsNotNone(mm._channel)
            for name in ["ram", "dino", "sam", "clip"]:
                self.assertIsNotNone(mm.models[name].server_channel)
            mm.close()

    @patch("tools.model_manager.server.ModelProcess.start")
    def test_local_init_no_cuda_mps_uses_mps_device(self, mock_start):
        from tools.model_manager import ModelManager
        with patch("torch.cuda.device_count", return_value=0), \
             patch("torch.backends.mps.is_available", return_value=True), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            mm = ModelManager(device="mps", local=True)
            process = mm._processes[0]
            self.assertEqual(process._device_override, "mps")
            mm.close()

    @patch("tools.model_manager.server.ModelProcess.start")
    def test_local_init_no_cuda_subset_excludes_embedding_completion(self, mock_start):
        from tools.model_manager import ModelManager
        with patch("torch.cuda.device_count", return_value=0), \
             patch("torch.backends.mps.is_available", return_value=False), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
            mm = ModelManager(device="cpu", local=True)
            process = mm._processes[0]
            self.assertIn("ram", process.model_subset)
            self.assertNotIn("embedding", process.model_subset)
            self.assertNotIn("completion", process.model_subset)
            mm.close()


# ═════════════════════════════════════════════════════════════════════════════
# Integration: ProcessChannel round-trip through a real worker thread
# (no subprocess, tests the channel logic end-to-end without spawn overhead)
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessChannelIntegration(unittest.TestCase):
    def setUp(self):
        self.ch = _make_channel()
        self._stop = threading.Event()

    def tearDown(self):
        self._stop.set()
        self.ch.shutdown()

    def _worker(self, model_subset, transform):
        """Minimal server loop running in a thread."""
        import queue as q_mod
        while not self._stop.is_set():
            with self.ch.cond_s:
                found = False
                for url, q in self.ch.input.items():
                    if model_subset and all(m not in url for m in model_subset):
                        continue
                    try:
                        item = q.get_nowait()
                    except q_mod.Empty:
                        continue
                    found = True
                    if url in self.ch.UNBATCHED:
                        req_id, (args, kwargs) = item
                        self.ch.put_s((req_id, transform(args[0])))
                    else:
                        req_id, (args, kwargs) = item
                        self.ch.put_s([(req_id, transform(args[0]))])
                    break
                if not found:
                    self.ch.cond_s.wait(timeout=0.01)

    def test_end_to_end_ram(self):
        t = threading.Thread(target=self._worker, args=(["ram"], str.upper), daemon=True)
        t.start()
        self.ch.put_c(1, "/ram", (("hello",), {}))
        result = self.ch.get_c(1)
        self.assertEqual(result, "HELLO")

    def test_end_to_end_sam_unbatched(self):
        t = threading.Thread(target=self._worker, args=(["sam"], lambda x: x + "_mask"), daemon=True)
        t.start()
        self.ch.put_c(2, "/sam", (("img",), {}))
        result = self.ch.get_c(2)
        self.assertEqual(result, "img_mask")


# ═════════════════════════════════════════════════════════════════════════════
# Integration: full ModelManager + real CLIP inference
# Run with: pytest tests/test_model_manager.py -v -m integration
# ═════════════════════════════════════════════════════════════════════════════

# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mm():
    from tools.model_manager import ModelManager
    manager = ModelManager(local=True)
    yield manager
    manager.close()


# ═════════════════════════════════════════════════════════════════════════════
# Integration tests
# Run with: pytest tests/test_model_manager.py -v -m integration
# ═════════════════════════════════════════════════════════════════════════════

import numpy as _np


def _rand_rgb(h, w):
    return _np.random.randint(0, 256, (h, w, 3), dtype=_np.uint8)


@pytest.mark.integration
class TestCLIPIntegration:
    def test_predict_image_returns_correct_shape(self, mm):
        feat = mm.get_model("clip").predict_image(_rand_rgb(224, 224))
        assert feat.shape == (512,)

    def test_predict_image_is_normalized(self, mm):
        feat = mm.get_model("clip").predict_image(_rand_rgb(224, 224), normalize=True)
        assert abs(float(_np.linalg.norm(feat)) - 1.0) < 1e-5

    def test_predict_image_unnormalized_nonunit(self, mm):
        feat = mm.get_model("clip").predict_image(_rand_rgb(224, 224), normalize=False)
        norm = float(_np.linalg.norm(feat))
        assert norm > 0.0
        assert abs(norm - 1.0) > 0.01

    def test_predict_text_returns_correct_shape(self, mm):
        feat = mm.get_model("clip").predict_text(["a cat", "a dog"])
        assert feat.shape == (2, 512)

    def test_image_text_similarity_is_reasonable(self, mm):
        clip = mm.get_model("clip")
        red_img = _np.zeros((224, 224, 3), dtype=_np.uint8)
        red_img[:, :, 0] = 255
        img_feat = clip.predict_image(red_img, normalize=True)
        text_feats = clip.predict_text(["a red image", "a blue image"])
        assert float(_np.dot(img_feat, text_feats[0])) > float(_np.dot(img_feat, text_feats[1]))


@pytest.mark.integration
class TestRAMIntegration:
    def test_predict_single_returns_list_of_strings(self, mm):
        tags = mm.get_model("ram").predict(_rand_rgb(384, 384))
        assert isinstance(tags, list)
        assert all(isinstance(t, str) for t in tags)

    def test_predict_concurrent_both_get_results(self, mm):
        # RAMClient batches concurrent single-image requests server-side.
        # Sending a list directly through the channel double-wraps args and crashes.
        import threading
        results = {}
        def call(key):
            results[key] = mm.get_model("ram").predict(_rand_rgb(384, 384))
        threads = [threading.Thread(target=call, args=(i,)) for i in range(2)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=120)
        assert len(results) == 2
        for tags in results.values():
            assert isinstance(tags, list)
            assert all(isinstance(t, str) for t in tags)

    def test_predict_gradient_image_returns_list(self, mm):
        img = _np.zeros((384, 384, 3), dtype=_np.uint8)
        img[:, :, 0] = _np.linspace(0, 255, 384, dtype=_np.uint8)
        tags = mm.get_model("ram").predict(img)
        assert isinstance(tags, list)


@pytest.mark.integration
class TestDINOIntegration:
    def test_predict_returns_boxes_and_phrases(self, mm):
        img = _rand_rgb(512, 512)
        boxes, phrases = mm.get_model("dino").predict(img, ["cat", "dog"])
        assert isinstance(boxes, _np.ndarray)
        assert isinstance(phrases, list)
        assert boxes.ndim == 2 and boxes.shape[1] == 4

    def test_predict_empty_tags_returns_empty(self, mm):
        img = _rand_rgb(512, 512)
        boxes, phrases = mm.get_model("dino").predict(img, [])
        assert boxes.shape[0] == 0
        assert phrases == []

    def test_predict_boxes_within_image_bounds(self, mm):
        h, w = 512, 512
        img = _rand_rgb(h, w)
        boxes, phrases = mm.get_model("dino").predict(img, ["building", "person", "car"])
        if boxes.shape[0] > 0:
            assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= w).all()
            assert (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= h).all()

    def test_predict_annotate_returns_three_values(self, mm):
        img = _rand_rgb(512, 512)
        result = mm.get_model("dino").predict(img, ["cat"], annotate=True)
        assert len(result) == 3
        boxes, phrases, annotated = result
        assert annotated.shape == img.shape
        assert annotated.dtype == _np.uint8

    def test_predict_boxes_and_phrases_same_length(self, mm):
        img = _rand_rgb(512, 512)
        boxes, phrases = mm.get_model("dino").predict(img, ["building", "person"])
        assert len(phrases) == boxes.shape[0]


@pytest.mark.integration
class TestSAMIntegration:
    def test_predict_single_box_returns_mask(self, mm):
        img = _rand_rgb(256, 256)
        boxes = _np.array([[32, 32, 128, 128]], dtype=_np.float32)
        masks = mm.get_model("sam").predict(img, boxes)
        assert masks.shape == (1, img.shape[0], img.shape[1])
        assert masks.dtype == bool

    def test_predict_multiple_boxes_returns_multiple_masks(self, mm):
        boxes = _np.array([[10, 10, 80, 80], [100, 100, 200, 200], [50, 50, 150, 150]], dtype=_np.float32)
        masks = mm.get_model("sam").predict(_rand_rgb(256, 256), boxes)
        assert masks.shape[0] == 3

    def test_predict_annotate_returns_masks_and_rgb(self, mm):
        img = _rand_rgb(256, 256)
        boxes = _np.array([[32, 32, 128, 128]], dtype=_np.float32)
        masks, ann = mm.get_model("sam").predict(img, boxes, annotate=True)
        assert masks.shape[0] == 1
        assert ann.shape == img.shape
        assert ann.dtype == _np.uint8

    def test_predict_mask_values_are_boolean(self, mm):
        boxes = _np.array([[0, 0, 256, 256]], dtype=_np.float32)
        masks = mm.get_model("sam").predict(_rand_rgb(256, 256), boxes)
        assert set(masks.flatten().tolist()).issubset({True, False})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
