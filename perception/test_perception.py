"""Offline tests: no robot, no Nebius calls.  Run: .venv/bin/python -m unittest -v perception/test_perception.py"""
import base64
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import camera  # noqa: E402
import nebius_vision  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-body"


class DecodeTests(unittest.TestCase):
    def test_base64_jpeg(self):
        msg = {"format": "jpeg", "data": base64.b64encode(JPEG).decode()}
        self.assertEqual(camera.decode_compressed_image(msg), JPEG)

    def test_int_list_jpeg(self):
        self.assertEqual(camera.decode_compressed_image({"format": "rgb8; jpeg compressed bgr8", "data": list(JPEG)}), JPEG)

    def test_rejects_non_jpeg_format(self):
        with self.assertRaises(ValueError):
            camera.decode_compressed_image({"format": "h264", "data": ""})

    def test_rejects_non_jpeg_bytes(self):
        with self.assertRaises(ValueError):
            camera.decode_compressed_image({"format": "jpeg", "data": base64.b64encode(b"PNG...").decode()})

    def test_unknown_camera(self):
        with self.assertRaises(ValueError):
            camera.grab_frames(["nope"])


class MessageTests(unittest.TestCase):
    def test_labels_images_and_puts_prompt_last(self):
        msgs = nebius_vision.build_messages({"head": JPEG, "wrist": JPEG}, "What is there?", system="sys")
        self.assertEqual(msgs[0], {"role": "system", "content": "sys"})
        content = msgs[1]["content"]
        self.assertEqual([c["type"] for c in content], ["text", "image_url", "text", "image_url", "text"])
        self.assertEqual(content[0]["text"], "Image from: head")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[-1]["text"], "What is there?")

    def test_rejects_empty_prompt_and_no_images(self):
        with self.assertRaises(ValueError):
            nebius_vision.build_messages({"head": JPEG}, "  ")
        with self.assertRaises(ValueError):
            nebius_vision.build_messages({}, "hi")


class AskTests(unittest.TestCase):
    def test_parses_reply(self):
        reply = {"model": "m", "choices": [{"message": {"content": " a book "}}], "usage": {"prompt_tokens": 5}}
        with mock.patch.object(nebius_vision, "_request", return_value=reply) as req:
            out = nebius_vision.ask({"head": JPEG}, "what?", model="m", key="k")
        self.assertEqual(out["text"], "a book")
        path, key, payload = req.call_args.args
        self.assertEqual((path, key, payload["model"]), ("/chat/completions", "k", "m"))

    def test_thinking_off_by_default(self):
        p = nebius_vision.build_payload({"head": JPEG}, "what?", "m")
        self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(p["max_tokens"], 800)
        p = nebius_vision.build_payload({"head": JPEG}, "what?", "m", think=True)
        self.assertNotIn("chat_template_kwargs", p)
        self.assertEqual(p["max_tokens"], 2000)

    def test_empty_answer_is_an_error(self):
        reply = {"choices": [{"message": {"content": None, "reasoning_content": "..."}, "finish_reason": "length"}]}
        with mock.patch.object(nebius_vision, "_request", return_value=reply):
            with self.assertRaisesRegex(RuntimeError, "empty answer.*length"):
                nebius_vision.ask({"head": JPEG}, "what?", key="k")

    def test_bad_shape_raises(self):
        with mock.patch.object(nebius_vision, "_request", return_value={"oops": 1}):
            with self.assertRaises(RuntimeError):
                nebius_vision.ask({"head": JPEG}, "what?", key="k")

    def test_missing_key_raises(self):
        with mock.patch.dict("os.environ", {"NEBIUS_API_KEY": ""}), \
             mock.patch.object(nebius_vision, "KEY_FILE", Path("/nonexistent/key")):
            with self.assertRaises(RuntimeError):
                nebius_vision.load_key()


if __name__ == "__main__":
    unittest.main()
