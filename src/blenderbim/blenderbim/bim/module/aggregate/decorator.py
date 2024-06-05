# BlenderBIM Add-on - OpenBIM Blender Add-on
# Copyright (C) 2023 Dion Moult <dion@thinkmoult.com>
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

import gpu
import bmesh
import ifcopenshell
import blenderbim.tool as tool
from bpy.types import SpaceView3D
from mathutils import Vector
from gpu_extras.batch import batch_for_shader


class AggregateDecorator:
    installed = None

    @classmethod
    def install(cls, context):
        if cls.installed:
            cls.uninstall()
        handler = cls()
        cls.installed = SpaceView3D.draw_handler_add(handler, (context,), "WINDOW", "POST_VIEW")

    @classmethod
    def uninstall(cls):
        try:
            SpaceView3D.draw_handler_remove(cls.installed, "WINDOW")
        except ValueError:
            pass
        cls.installed = None

    def draw_batch(self, shader_type, content_pos, color, indices=None):
        shader = self.line_shader if shader_type == "LINES" else self.shader
        batch = batch_for_shader(shader, shader_type, {"pos": content_pos}, indices=indices)
        shader.uniform_float("color", color)
        batch.draw(shader)

    def __call__(self, context):
        self.addon_prefs = context.preferences.addons["blenderbim"].preferences
        theme = context.preferences.themes.items()[0][1]
        # selected_elements_color = self.addon_prefs.decorator_color_selected
        selected_element_color = (*theme.view_3d.object_active, 1) #unwrap color values and adds alpha=1
        unselected_elements_color = self.addon_prefs.decorator_color_unselected
        special_elements_color = self.addon_prefs.decorator_color_special

        def transparent_color(color, alpha=0.1):
            color = [i for i in color]
            color[3] = alpha
            return color

        
        def select_aggregate_objects(element, objs = []):
            obj = tool.Ifc.get_object(element)
            # obj.select_set(True)
            parts = ifcopenshell.util.element.get_parts(element)
            if parts:
                for part in parts:
                    if part.is_a("IfcElementAssembly"):
                        select_objects_and_add_data(part, objs)
                    obj = tool.Ifc.get_object(part)
                    objs.append(obj)
            objs.append(obj)
            return objs
                    

        gpu.state.point_size_set(6)
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        gpu.state.point_size_set(6)

        self.line_shader = gpu.shader.from_builtin("3D_POLYLINE_UNIFORM_COLOR")
        self.line_shader.bind()  # required to be able to change uniforms of the shader
        # POLYLINE_UNIFORM_COLOR specific uniforms
        self.line_shader.uniform_float("viewportSize", (context.region.width, context.region.height))
        self.line_shader.uniform_float("lineWidth", 2.0)

        # general shader
        self.shader = gpu.shader.from_builtin("3D_UNIFORM_COLOR")

        ######################################################################

        # Get the selected objects
        selected_objects = context.selected_objects
        if not selected_objects:
            return {"FINISHED"}
        selected_element = tool.Ifc.get_entity(selected_objects[0])
        if selected_element.is_a("IfcElementAssembly"):
            pass
        elif selected_element.Decomposes:
            if selected_element.Decomposes[0].RelatingObject.is_a("IfcElementAssembly"):
                selected_element = selected_element.Decomposes[0].RelatingObject
                selected_obj = tool.Ifc.get_object(selected_element)
            else:
                self.report({"INFO"}, "Object is not part of a IfcElementAssembly.")
                return {"FINISHED"}
            
        selected_objects = select_aggregate_objects(selected_element)
                    
        # Initialize variables to store the min and max coordinates of the bounding box
        min_coords = Vector((float('inf'), float('inf'), float('inf')))
        max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

        
        selected_vertices = []
        selected_edges = []
        selected_tris = []
        
        
        for obj in selected_objects:
            if obj.type == 'MESH':
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                obj.data.calc_loop_triangles()

                offset = len(selected_vertices)
                selected_vertices.extend([tuple(obj.matrix_world @ v.co) for v in bm.verts])
                selected_edges.extend([tuple([v.index + offset for v in e.verts]) for e in bm.edges])
                selected_tris.extend([tuple([i + offset for i in t.vertices]) for t in obj.data.loop_triangles])
            
                # Iterate through the selected objects and calculate their bounding box
                # Get the world matrix of the object
                world_matrix = obj.matrix_world

                # Get the object's vertices in world coordinates
                vertices = [world_matrix @ vertex.co for vertex in obj.data.vertices]

                # Update min and max coordinates
                for vertex in vertices:
                    min_coords = Vector((min(vertex[i], min_coords[i]) for i in range(3)))
                    max_coords = Vector((max(vertex[i], max_coords[i]) for i in range(3)))

        # Offset the bounding box limits to avoid z-fighting
        offset = 0.1
        min_coords = Vector([value-offset for value in min_coords])
        max_coords = Vector([value+offset for value in max_coords])
        
        # Calculate the eight vertices of the bounding box
        vertices = [
            Vector((min_coords.x, min_coords.y, min_coords.z)),
            Vector((min_coords.x, min_coords.y, max_coords.z)),
            Vector((min_coords.x, max_coords.y, min_coords.z)),
            Vector((min_coords.x, max_coords.y, max_coords.z)),
            Vector((max_coords.x, min_coords.y, min_coords.z)),
            Vector((max_coords.x, min_coords.y, max_coords.z)),
            Vector((max_coords.x, max_coords.y, min_coords.z)),
            Vector((max_coords.x, max_coords.y, max_coords.z))
        ]


        lines_indices = (
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7)
        )

        tris_indices = (
            (0, 1, 3), (0, 2, 3), (4, 5, 7), (4, 6, 7),
            (0, 4, 5), (0, 1, 5), (2, 0, 4), (2, 6, 4),
            (1, 3, 7), (1, 5, 7), (3, 2, 6), (3, 6, 7)
        )
        
        self.draw_batch("LINES", vertices, selected_element_color, lines_indices)
        self.draw_batch("TRIS", vertices, transparent_color((1, 1, 1, 1)), tris_indices)
