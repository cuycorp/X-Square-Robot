"""Offline tests for the web UI: fake camera hub, no robot, no Nebius calls.
Run: .venv/bin/python -m unittest -v perception/test_webui.py"""
import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import webui  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0fake"


class FakeHub:
    def __init__(self, age=0.1):
        self.frames = {"head": (JPEG, time.monotonic() - age, 1)}
        self.connected, self.last_error, self.wanted = True, "", set()

    def want(self, name):
        if name not in webui.camera.CAMERAS:
            raise KeyError(name)
        self.wanted.add(name)

    def latest(self, name):
        return self.frames.get(name)

    def wait_newer(self, name, seq, timeout=2.0):
        return self.frames.get(name)


REPLY = {"text": "A book and a pen.", "model": "m", "usage": {"prompt_tokens": 9, "completion_tokens": 5}}


class AskAboutTests(unittest.TestCase):
    def test_answers_with_the_frame_it_used(self):
        with mock.patch.object(webui.nebius_vision, "ask", return_value=REPLY) as ask:
            out = webui.ask_about(FakeHub(), " What is there? ", ["head"], False, "m")
        self.assertEqual(out["answer"], "A book and a pen.")
        self.assertEqual(out["cameras"], ["head"])
        self.assertIn("head", out["frames"])
        images, question = ask.call_args.args
        self.assertEqual(question, "What is there?")
        self.assertEqual(list(images.values()), [JPEG])

    def test_rejects_empty_and_long_questions(self):
        for q in ("", "   ", "x" * (webui.MAX_QUESTION_CHARS + 1)):
            with self.assertRaises(ValueError):
                webui.ask_about(FakeHub(), q, ["head"], False, "m")

    def test_refuses_stale_frame(self):
        with self.assertRaisesRegex(RuntimeError, "No fresh frame"):
            webui.ask_about(FakeHub(age=webui.MAX_FRAME_AGE_S + 5), "hi", ["head"], False, "m")

    def test_unknown_cameras_fall_back_to_head(self):
        with mock.patch.object(webui.nebius_vision, "ask", return_value=REPLY):
            out = webui.ask_about(FakeHub(), "hi", ["nope"], False, "m")
        self.assertEqual(out["cameras"], ["head"])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), webui.make_handler(FakeHub(), "test-model"))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def post(self, body):
        req = urllib.request.Request(self.base + "/ask", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index_page(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            self.assertIn(b"Robot Eyes", r.read())

    def test_status(self):
        with urllib.request.urlopen(self.base + "/status", timeout=5) as r:
            s = json.loads(r.read())
        self.assertTrue(s["connected"])
        self.assertEqual(s["model"], "test-model")
        self.assertIn("head", s["cameras"])

    def test_ask_ok(self):
        with mock.patch.object(webui.nebius_vision, "ask", return_value=REPLY):
            code, body = self.post({"question": "What is there?", "cameras": ["head"]})
        self.assertEqual((code, body["answer"]), (200, "A book and a pen."))

    def test_ask_empty_question_is_400(self):
        code, body = self.post({"question": ""})
        self.assertEqual(code, 400)
        self.assertIn("question", body["error"].lower())

    def test_unknown_stream_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.base + "/stream/nope", timeout=5)
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
