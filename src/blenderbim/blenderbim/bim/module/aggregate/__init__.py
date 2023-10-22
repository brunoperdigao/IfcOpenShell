# BlenderBIM Add-on - OpenBIM Blender Add-on
# Copyright (C) 2020, 2021 Dion Moult <dion@thinkmoult.com>
#
# This file is part of BlenderBIM Add-on.
#
# BlenderBIM Add-on is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BlenderBIM Add-on is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BlenderBIM Add-on.  If not, see <http://www.gnu.org/licenses/>.

import bpy
from . import ui, prop, operator

classes = (
    operator.BIM_OT_assign_object,
    operator.BIM_OT_unassign_object,
    operator.BIM_OT_enable_editing_aggregate,
    operator.BIM_OT_disable_editing_aggregate,
    operator.BIM_OT_add_aggregate,
    operator.BIM_OT_select_parts,
    operator.BIM_OT_select_aggregate,
    operator.BIM_OT_add_part_to_object,
    operator.BIM_OT_select_all_objects_in_aggregate,
    operator.BIM_OT_enable_aggregate_decorator,
    prop.BIMObjectAggregateProperties,
    ui.BIM_PT_aggregate,
)

addon_keymaps = []

def register():
    bpy.types.Object.BIMObjectAggregateProperties = bpy.props.PointerProperty(type=prop.BIMObjectAggregateProperties)
    bpy.types.Scene.aggregate_decorator = bpy.props.BoolProperty(name = "Aggregate Decorator", default=True)
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.new(name="Object Mode", space_type="EMPTY")
        kmi = km.keymap_items.new("bim.select_all_objects_in_aggregate", "LEFTMOUSE", "DOUBLE_CLICK")
        addon_keymaps.append((km, kmi))


def unregister():
    del bpy.types.Object.BIMObjectAggregateProperties
    del bpy.types.Scene.aggregate_decorator
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        for km, kmi in addon_keymaps:
            km.keymap_items.remove(kmi)
    addon_keymaps.clear()
