# This file was generated with the assistance of an AI coding tool.
import time

import bpy
import pytest

import bonsai.tool as tool


def run_iter_from_timer(event_iter):
    i = iter(event_iter)

    def event_step():
        ret = next(i, None)
        if ret is None:
            return None
        return 0.0

    bpy.app.timers.register(event_step, first_interval=0.0)


def preset_event_simulate(window, event_type, value, x=960, y=540):
    if value == "TAP":
        yield window.event_simulate(event_type, "PRESS", x=x, y=y)
        yield window.event_simulate(event_type, "RELEASE", x=x, y=y)
    else:
        yield window.event_simulate(event_type, value, x=x, y=y)


@pytest.fixture
def window():
    yield bpy.context.window
    try:
        bpy.app.use_event_simulate = False
    except ValueError:
        pass


def measure_tool_test(window):
    try:
        yield from preset_event_simulate(window, "ESC", "TAP")

        measure_settings = tool.Project.get_measure_tool_settings()
        measure_settings.measurement_type = "POLYLINE"
        for obj in tool.Blender.get_selected_objects():
            obj.select_set(False)
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        with bpy.context.temp_override(
                            area=area, region=region, space_data=area.spaces[0]
                        ):
                            bpy.ops.bim.measure_tool("INVOKE_DEFAULT", measure_type="POLYLINE")
                break

        yield from preset_event_simulate(window, "MOUSEMOVE", "NOTHING", x=960, y=540)
        yield from preset_event_simulate(window, "LEFTMOUSE", "TAP", x=960, y=540)

        snap_point = tool.Model.get_polyline_props().snap_mouse_point[0]
        assert snap_point.snap_object is not None, "First click should have a snap_object"

        yield from preset_event_simulate(window, "MOUSEMOVE", "NOTHING", x=860, y=540)
        yield from preset_event_simulate(window, "MOUSEMOVE", "NOTHING", x=860, y=540)
        yield from preset_event_simulate(window, "MOUSEMOVE", "NOTHING", x=860, y=540)
        yield from preset_event_simulate(window, "MOUSEMOVE", "NOTHING", x=860, y=540)
        time.sleep(1)
        yield from preset_event_simulate(window, "LEFTMOUSE", "TAP", x=860, y=540)
        snap_point = tool.Model.get_polyline_props().snap_mouse_point[0]
        assert snap_point.snap_object is not None, "Second click should have a snap_object"
        time.sleep(1)
        yield from preset_event_simulate(window, "RET", "TAP")

        polyline_props = tool.Model.get_polyline_props()
        assert len(polyline_props.measurement_polyline) > 0, "Should have measurement polylines"

        measure_poly = polyline_props.measurement_polyline[-1]
        num_points = len(measure_poly.polyline_points)
        assert num_points == 2, "Should have 2 points in polyline"
    finally:
        bpy.app.use_event_simulate = False
        bpy.ops.wm.quit_blender()


@pytest.mark.modal
def test_measure_tool(window):
    run_iter_from_timer(measure_tool_test(window))


if __name__ == "__main__":
    window = bpy.context.window
    run_iter_from_timer(measure_tool_test(window))
