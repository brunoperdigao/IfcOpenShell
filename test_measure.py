# test_measure.py
import time

import bonsai.tool as tool
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


def measure_tool_test(window):
    def preset_event_simulate(window, event_type, value, x=960, y=540):
        if value == 'TAP':
            yield window.event_simulate(event_type, 'PRESS', x=x, y=y)
            yield window.event_simulate(event_type, 'RELEASE', x=x, y=y)
        else:
            yield window.event_simulate(event_type, value, x=x, y=y)

    yield from preset_event_simulate(window, 'ESC', 'TAP')

    # bpy.ops.wm.tool_set_by_id(name="bim.explore_tool")


    measure_settings = tool.Project.get_measure_tool_settings()
    measure_settings.measurement_type = "POLYLINE"
    for obj in tool.Blender.get_selected_objects():
        obj.select_set(False)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    with bpy.context.temp_override(area=area, region=region, space_data=area.spaces[0]):
                        bpy.ops.bim.measure_tool("INVOKE_DEFAULT", measure_type="POLYLINE")
            break
    # bpy.ops.bim.measure_tool("INVOKE_DEFAULT", measure_type="POLYLINE")

    yield from preset_event_simulate(window, 'MOUSEMOVE', 'NOTHING', x=960, y=540)
    yield from preset_event_simulate(window, 'LEFTMOUSE', 'TAP', x=960, y=540)

    snap_point = tool.Model.get_polyline_props().snap_mouse_point[0]
    print(f"After first tap - snap_object: '{snap_point.snap_object}', snap_type: '{snap_point.snap_type}'")

    yield from preset_event_simulate(window, 'MOUSEMOVE', 'NOTHING', x=860, y=540)
    yield from preset_event_simulate(window, 'MOUSEMOVE', 'NOTHING', x=860, y=540)
    yield from preset_event_simulate(window, 'MOUSEMOVE', 'NOTHING', x=860, y=540)
    yield from preset_event_simulate(window, 'MOUSEMOVE', 'NOTHING', x=860, y=540)
    time.sleep(1)
    yield from preset_event_simulate(window, 'LEFTMOUSE', 'TAP', x=860, y=540)
    snap_point = tool.Model.get_polyline_props().snap_mouse_point[0]
    print(f"After second tap - snap_object: '{snap_point.snap_object}', snap_type: '{snap_point.snap_type}'")
    time.sleep(1)
    yield from preset_event_simulate(window, 'RET', 'TAP')

    polyline_props = tool.Model.get_polyline_props()
    print(f"Number of measurement polylines: {len(polyline_props.measurement_polyline)}")
    if polyline_props.measurement_polyline:
        measure_poly = polyline_props.measurement_polyline[-1]
        num_points = len(measure_poly.polyline_points)
        num_edges = num_points - 1  # Edges are computed dynamically, not stored
        print(f"Points in polyline: {num_points}")
        print(f"Edges in polyline: {num_edges}")


    bpy.app.use_event_simulate = False


if __name__ == '__main__':
    window = bpy.context.window
    run_iter_from_timer(measure_tool_test(window))
