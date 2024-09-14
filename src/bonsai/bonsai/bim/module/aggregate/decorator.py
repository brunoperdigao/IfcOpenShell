# Bonsai - OpenBIM Blender Add-on
# Copyright (C) 2024 Bruno Perdigão <contact@brunopo.com>
#
# This file is part of Bonsai.
#
# Bonsai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Bonsai is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Bonsai.  If not, see <http://www.gnu.org/licenses/>.
import bpy
import gpu
import bmesh
import ifcopenshell
import bonsai.tool as tool
import numpy as np
from bpy.types import SpaceView3D
from mathutils import Vector
from gpu_extras.batch import batch_for_shader

class AggregateDecorator:
    is_installed = True
    handlers = []

    @classmethod
    def install(cls, context):
        if cls.is_installed:
            cls.uninstall()
        handler = cls()
        cls.handlers.append(SpaceView3D.draw_handler_add(handler, (context,), "WINDOW", "POST_VIEW"))
        # cls.handlers.append(SpaceView3D.draw_handler_add(handler.draw_outline, (context,), "WINDOW", "POST_VIEW"))
        cls.is_installed = True

    @classmethod
    def uninstall(cls):
        for handler in cls.handlers:
            try:
                SpaceView3D.draw_handler_remove(handler, "WINDOW")
            except ValueError:
                pass
        cls.is_installed = False

    def draw_batch_custom_shader(self, obj, content_pos, normals, color, indices=None):

        # Shader from https://blender.stackexchange.com/a/274460

        vertex_shader = '''
in vec3 position;
in vec3 normal;
in vec4 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 camera_location;
uniform float factor1;
uniform float factor2;

out vec4 fcolor;
void main()
{

    vec3 pos = vec3(model * vec4(position, 1.0));
    vec3 nor = mat3(transpose(inverse(model))) * normal;

    float d = distance(pos, camera_location) * factor1;
    vec3 offset = nor * vec3(d);
    vec3 p = pos + offset;
    //p = pos;

    // hmmm?
    //vec3 n = nor * vec3(0.0);
    //nor = nor * vec3(0.0);

    vec3 dir = p - camera_location;
    dir = normalize(dir) * vec3(factor2);
    p = p + dir;

    //gl_Position = projection * view * model * vec4(position, 1.0);
    //gl_Position = projection * view * model * vec4(p, 1.0);
    gl_Position = projection * view * vec4(p, 1.0);
    //fcolor = color * vec4(offset, 1.0);
    fcolor = color;
    //fcolor = color * vec4(dir / vec3(factor2), 1.0);
}
'''
        fragment_shader = '''
in vec4 fcolor;
out vec4 fragColor;
void main()
{
    fragColor = blender_srgb_to_framebuffer_space(fcolor);
}
'''
        shader = gpu.types.GPUShader(vertex_shader, fragment_shader, )
        batch = batch_for_shader(shader, 'TRIS', {"position": content_pos, "normal": normals, "color": color}, indices=indices)

        gpu.state.depth_test_set('LESS')
    
        shader.bind()
        shader.uniform_float("model", obj.matrix_world)
        shader.uniform_float("view", bpy.context.region_data.view_matrix)
        shader.uniform_float("projection", bpy.context.region_data.window_matrix)
    
        cl = bpy.context.region_data.view_matrix.inverted().translation
        shader.uniform_float("camera_location", cl)
        shader.uniform_float("factor1", 0.003)
        shader.uniform_float("factor2", 50.0)
    
        batch.draw(shader)
    
        gpu.state.depth_test_set('NONE')

        # shader.uniform_float("color", color)

    def draw_batch(self, shader_type, content_pos, color, indices=None):
        shader = self.line_shader if shader_type == "LINES" else self.shader
        batch = batch_for_shader(shader, shader_type, {"pos": content_pos}, indices=indices)
        shader.uniform_float("color", color)
        batch.draw(shader)

    def draw_outline(self, context, objs):
        for obj in objs:
            if obj in context.selected_objects:
                continue
            me = obj.data
            me.calc_loop_triangles()
        
            vs = np.zeros((len(me.vertices) * 3, ), dtype=np.float32, )
            me.vertices.foreach_get('co', vs)
            vs.shape = (-1, 3, )
            ns = np.zeros((len(me.vertices) * 3, ), dtype=np.float32, )
            me.vertices.foreach_get('normal', ns)
            ns.shape = (-1, 3, )
            fs = np.zeros((len(me.loop_triangles) * 3, ), dtype=np.int32, )
            me.loop_triangles.foreach_get('vertices', fs)
            fs.shape = (-1, 3, )
            cs = np.full((len(me.vertices), 4), (1.0, 1.0, 1.0, 1.0), dtype=np.float32, )
            self.draw_batch_custom_shader(obj, vs, ns, cs, indices=fs)
        
        
        
    def __call__(self, context):
        print(self.is_installed)
        self.addon_prefs = tool.Blender.get_addon_preferences()
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
                    obj = tool.Ifc.get_object(part)
                    objs.append(obj)
            objs.append(obj)
            return objs


        gpu.state.point_size_set(4)
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)

        self.line_shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        self.line_shader.bind()  # required to be able to change uniforms of the shader
        # POLYLINE_UNIFORM_COLOR specific uniforms
        self.line_shader.uniform_float("viewportSize", (context.region.width, context.region.height))
        self.line_shader.uniform_float("lineWidth", 0.5)

        # general shader
        self.shader = gpu.shader.from_builtin("UNIFORM_COLOR")


        # TODO Refactor to use the api to get the aggregate
        # Get the selected objects
        selected_objects = context.selected_objects
        if not selected_objects:
            return
        selected_element = tool.Ifc.get_entity(selected_objects[0])
        if selected_element.is_a("IfcElementAssembly"):
            pass
        if selected_element.Decomposes:
            if selected_element.Decomposes[0].RelatingObject.is_a("IfcElementAssembly"):
                selected_element = selected_element.Decomposes[0].RelatingObject
            else:
                return
        else:
            return

        filtered_objects = select_aggregate_objects(selected_element)
        self.draw_outline(context, filtered_objects)

        # Initialize variables to store the min and max coordinates of the bounding box
        min_coords = Vector((float('inf'), float('inf'), float('inf')))
        max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))


        selected_vertices = []
        selected_edges = []
        selected_tris = []


        for obj in filtered_objects:
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
                bm.free()

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

        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        self.draw_batch("LINES", vertices, (1, 1, 1, 1), lines_indices)
        # self.draw_batch("TRIS", vertices, transparent_color((1, 1, 1, 1)), tris_indices)
