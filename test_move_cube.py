# test_move_cube.py
import time

import bpy


def run_iter_from_timer(event_iter):
    i = iter(event_iter)

    SPEED = 0.0

    def event_step():
        ret = next(i, None)
        if ret is None:
            return None
        return 0.0

    bpy.app.timers.register(event_step, first_interval=0.0)


def select_move_deselect_cube(window):

    def preset_event_simulate(window, event_type, value, x=960, y=540):
        if value == 'TAP':
            yield window.event_simulate(event_type, 'PRESS', x=x, y=y)
            yield window.event_simulate(event_type, 'RELEASE', x=x, y=y)
        else:
            yield window.event_simulate(event_type, value, x=x, y=y)

    yield from preset_event_simulate(window, 'ESC', 'TAP')
    # yield from preset_event_simulate(window, 'LEFTMOUSE', 'TAP', x=960, y=540)
    yield from preset_event_simulate(window, 'G', 'TAP', x=960, y=540)
    yield from preset_event_simulate(window, 'MOUSEMOVE', 'NOTHING', x=180, y=540)
    time.sleep(5)
    yield from preset_event_simulate(window, 'LEFTMOUSE', 'TAP', x=180, y=540)
    yield from preset_event_simulate(window, 'ESC', 'TAP')

    bpy.app.use_event_simulate = False


if __name__ == '__main__':
    window = bpy.context.window
    run_iter_from_timer(select_move_deselect_cube(window))
