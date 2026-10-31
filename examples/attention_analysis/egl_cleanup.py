"""Keep robosuite 1.4 rendering, readback and destruction on their owning context.

Hard reset can leave an old renderer in a reference cycle. Its finalizer must not
free MuJoCo GL objects with a newer renderer's context current on the same thread.
The caller's context is restored after cleanup; images and simulator state are unchanged.
"""

import os
from functools import wraps


def install_render_context_guard(renderer_type):
    """Bind before both operations; a policy/RPC callback need not preserve GL state.

    The simulator's existing render lock still covers render plus read_pixels.
    No additional render, simulator step or random draw is introduced.
    """
    changed = False
    for name in ("render", "read_pixels"):
        original = getattr(renderer_type, name)
        if getattr(original, "_starvla_context_guard", False):
            continue

        def bind_and_call(method):
            @wraps(method)
            def guarded(renderer, *args, **kwargs):
                context = getattr(renderer, "gl_ctx", None)
                if context is None or not getattr(context, "_context", None):
                    raise RuntimeError("Cannot render/read pixels from a missing or freed EGL context")
                context.make_current()  # Propagate binding failures before touching an output buffer.
                return method(renderer, *args, **kwargs)

            guarded._starvla_context_guard = True
            return guarded

        setattr(renderer_type, name, bind_and_call(original))
        changed = True
    return changed


def free_on_own_context(renderer, cleanup, egl):
    context = getattr(renderer, "gl_ctx", None)
    handle = getattr(context, "_context", None)
    if not handle:
        return cleanup(renderer)
    previous = egl.eglGetCurrentContext()
    display = egl.eglGetCurrentDisplay()
    draw, read = egl.eglGetCurrentSurface(egl.EGL_DRAW), egl.eglGetCurrentSurface(egl.EGL_READ)
    restore = bool(previous) and previous.address != handle.address
    context.make_current()
    try:
        return cleanup(renderer)
    finally:
        if restore and not egl.eglMakeCurrent(display, draw, read, previous):
            raise RuntimeError("Unable to restore EGL context after renderer cleanup")


def install_egl_cleanup():
    if os.environ.get("MUJOCO_GL") != "egl":
        return False
    from mujoco.egl import egl_ext as egl
    from robosuite.utils.binding_utils import MjRenderContext

    if os.environ.get("STARVLA_EGL_RENDER_GUARD", "1") != "0":
        install_render_context_guard(MjRenderContext)
    original = MjRenderContext.__del__
    if getattr(original, "_starvla_egl_cleanup", False):
        return True

    def cleanup(renderer):
        return free_on_own_context(renderer, original, egl)

    cleanup._starvla_egl_cleanup = True
    MjRenderContext.__del__ = cleanup
    return True
