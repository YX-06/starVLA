"""Regressions for deferred renderer destruction after a hard reset."""

from types import SimpleNamespace

import pytest

from examples.attention_analysis.egl_cleanup import free_on_own_context, install_render_context_guard


@pytest.mark.parametrize("same_context", [False, True])
def test_cleanup_binds_owner_and_restores_other_context(same_context):
    owner = SimpleNamespace(address=1)
    previous = owner if same_context else SimpleNamespace(address=2)
    state, calls = {"current": previous}, []

    def bind_owner():
        calls.append("bind_owner")
        state["current"] = owner

    def restore(display, draw, read, handle):
        assert (display, draw, read) == (3, 4, 5)
        calls.append("restore")
        state["current"] = handle
        return True

    egl = SimpleNamespace(
        EGL_DRAW="draw",
        EGL_READ="read",
        eglGetCurrentContext=lambda: state["current"],
        eglGetCurrentDisplay=lambda: 3,
        eglGetCurrentSurface=lambda kind: 4 if kind == "draw" else 5,
        eglMakeCurrent=restore,
    )
    renderer = SimpleNamespace(gl_ctx=SimpleNamespace(_context=owner, make_current=bind_owner))

    def cleanup(renderer):
        assert state["current"] is owner
        calls.append("free")
        state["current"] = None

    free_on_own_context(renderer, cleanup, egl)
    assert calls == (["bind_owner", "free"] if same_context else ["bind_owner", "free", "restore"])
    assert state["current"] is (None if same_context else previous)


def test_cleanup_of_already_freed_context():
    calls = []
    renderer = SimpleNamespace(gl_ctx=SimpleNamespace(_context=None))
    free_on_own_context(renderer, lambda value: calls.append(value), None)
    assert calls == [renderer]


def test_render_and_readback_rebind_owner_after_context_loss():
    state, calls = {"current": None}, []

    class Renderer:
        def __init__(self):
            self.gl_ctx = SimpleNamespace(_context=object(), make_current=self.bind)

        def bind(self):
            state["current"] = self.gl_ctx._context
            calls.append("bind")

        def render(self, value):
            assert state["current"] is self.gl_ctx._context
            calls.append("render")
            return value

        def read_pixels(self, *, width):
            assert state["current"] is self.gl_ctx._context
            calls.append("read")
            return width

    renderer = Renderer()
    assert install_render_context_guard(Renderer)
    assert not install_render_context_guard(Renderer)
    assert renderer.render(7) == 7
    state["current"] = None  # Simulate a callback or finalizer releasing the current thread's context.
    assert renderer.read_pixels(width=5) == 5
    assert calls == ["bind", "render", "bind", "read"]


def test_freed_context_and_binding_failure_never_reach_pixel_readback():
    calls = []

    class Renderer:
        def render(self):
            calls.append("render")

        def read_pixels(self):
            calls.append("read")

    install_render_context_guard(Renderer)
    renderer = Renderer()
    renderer.gl_ctx = SimpleNamespace(_context=None)
    with pytest.raises(RuntimeError, match="freed"):
        renderer.read_pixels()

    def fail_bind():
        raise RuntimeError("binding failed")

    renderer.gl_ctx = SimpleNamespace(_context=object(), make_current=fail_bind)
    with pytest.raises(RuntimeError, match="binding failed"):
        renderer.render()
    assert calls == []
