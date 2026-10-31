"""Real CPU websocket traffic must not mutate observations at 8-step chunk boundaries."""

import json
import threading

import numpy as np
from PIL import Image
from websockets.sync.server import serve

from deployment.model_server.tools import msgpack_numpy
from examples.LIBERO.eval_files import model2libero_interface as adapter


def test_websocket_chunks_resize_only_requests_and_do_not_mutate_input(tmp_path, monkeypatch):
    requests = []
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("STARVLA_TRACE_ALIGNMENT", str(tmp_path / "alignment.jsonl"))

    def policy(connection):
        connection.send(msgpack_numpy.packb({"action_chunk_size": 8}))
        for data in connection:
            query = msgpack_numpy.unpackb(data)
            requests.append(query)
            actions = np.arange(56, dtype=np.float32).reshape(1, 8, 7)
            connection.send(msgpack_numpy.packb({"data": {"actions": actions, "trace_id": f"trace{len(requests)}"}}))

    resizes = []
    original_resize = Image.Image.resize

    def resized(image, *args, **kwargs):
        resizes.append(image.size)
        return original_resize(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resized)
    with serve(policy, "127.0.0.1", 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = adapter.ModelClient(host="127.0.0.1", port=server.socket.getsockname()[1])
        try:
            client.reset("instruction", "libero_goal", 0, 0)
            for step in range(25):
                images = [np.full((256, 256, 3), step + camera, dtype=np.uint8) for camera in range(2)]
                originals = [a.copy() for a in images]
                response = client.step({"image": images, "lang": "instruction"}, step)
                for actual, expected in zip(images, originals, strict=True):
                    np.testing.assert_array_equal(actual, expected)
                assert client.last_inference_requested == (step % 8 == 0)
                np.testing.assert_array_equal(response["raw_action"]["world_vector"], np.arange(3) + 7 * (step % 8))
        finally:
            client.client.close()
            server.shutdown()
            thread.join(timeout=5)
    assert len(requests) == 4 and len(resizes) == 8
    assert [q["trace_context"]["env_step"] for q in requests] == [0, 8, 16, 24]
    for query in requests:
        assert [a.shape for a in query["examples"][0]["image"]] == [(224, 224, 3)] * 2
    rows = [json.loads(line) for line in (tmp_path / "alignment.jsonl").read_text().splitlines()]
    assert [row["action_query"] for row in rows] == [i % 8 for i in range(25)]
    assert [row["source_env_step"] for row in rows] == [8 * (i // 8) for i in range(25)]


def test_request_image_is_owned_even_when_already_at_target_size(monkeypatch):
    class Policy:
        def __init__(self, *args):
            pass

        def get_server_metadata(self):
            return {"action_chunk_size": 8}

        def predict_action(self, query):
            query["examples"][0]["image"][0][:] = 0
            return {"data": {"actions": np.zeros((1, 8, 7))}}

    monkeypatch.setattr(adapter, "WebsocketClientPolicy", Policy)
    client = adapter.ModelClient()
    image = np.full((224, 224, 3), 123, dtype=np.uint8)
    client.step({"image": [image, image], "lang": "instruction"}, 0)
    assert (image == 123).all()
