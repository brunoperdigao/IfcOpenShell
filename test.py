# test.py
import bpy


def run_iter_from_timer(event_iter):
    i = iter(event_iter)

    # latency between key presses, bigger numbers will make it run slower
    SPEED = 0.0

    def event_step():
        ret = next(i, None)
        if ret is None:
            return None
        return 0.0

    bpy.app.timers.register(event_step, first_interval=0.0)

def delete_default_cube(window):

    # template function to simplify things
    def preset_event_simulate(window, event_type, value, x=960, y=540):
        if value == 'TAP':  # just so you don't have to do both 'PRESS' and 'RELEASE'
            yield window.event_simulate(event_type, 'PRESS', x=x, y=y)
            yield window.event_simulate(event_type, 'RELEASE', x=x, y=y)
        else:
            yield window.event_simulate(event_type, value, x=x, y=y)

    yield from preset_event_simulate(window, 'ESC', 'TAP')  # close splash screen
    yield from preset_event_simulate(window, 'X', 'TAP')  # delete cube, opens confirmation prompt
    yield from preset_event_simulate(window, 'RET', 'TAP')  # confirms prompt

    # end event simulation, allow UI interaction by user
    bpy.app.use_event_simulate = False
    
    # optional, close Blender
    # import sys
    # sys.exit(0)


if __name__ == '__main__':
    window = bpy.context.window
    run_iter_from_timer(delete_default_cube(window))
