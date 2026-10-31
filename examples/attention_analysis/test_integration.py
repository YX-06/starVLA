"""Regression for actual checkpoint preparation failures."""

import subprocess

from examples.attention_analysis.prepare import download


def test_download_replaces_lfs_pointer_with_contents(tmp_path, monkeypatch):
    target = tmp_path / "tokenizer.json"
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abcd\nsize 1234\n"
    target.write_bytes(pointer)
    endpoints = []

    def fake_curl(command, **kwargs):
        endpoint = command[command.index("-o") - 1]
        endpoints.append(endpoint)
        output = command[command.index("-o") + 1]
        with open(output, "wb") as stream:
            stream.write(pointer if "/raw/" in endpoint else b'{"model": "actual tokenizer"}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_curl)
    download("https://huggingface.co/example/model/resolve/pinned/tokenizer.json", target)
    assert b"actual tokenizer" in target.read_bytes()
    assert "/raw/" in endpoints[0] and "/resolve/" in endpoints[1]
