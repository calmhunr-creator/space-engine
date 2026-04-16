#!/usr/bin/env python3
"""
Space Engineers-like 3D voxel sandbox with procedural GPU-driven world.
Uses ModernGL for GPU rendering and compute shaders for terrain generation.
"""

import moderngl
import numpy as np
import pygame
from pygame.locals import *
import math
import time

# Window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FOV = 60
NEAR = 0.1
FAR = 1000.0

# Voxel settings
VOXEL_SIZE = 1.0
CHUNK_SIZE = 16  # 16x16x16 voxels per chunk
RENDER_DISTANCE = 4  # chunks in each direction

# Camera settings
CAMERA_SPEED = 5.0
MOUSE_SENSITIVITY = 0.002


class Camera:
    def __init__(self, position=(0.0, 10.0, 0.0)):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
    def update_direction(self):
        front_x = math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        front_y = math.sin(math.radians(self.pitch))
        front_z = math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        self.front = np.array([front_x, front_y, front_z], dtype=np.float32)
        self.front = self.front / np.linalg.norm(self.front)
        self.right = np.cross(self.front, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        
    def get_view_matrix(self):
        return self.look_at(self.position, self.position + self.front, self.up)
    
    def look_at(self, eye, center, up):
        z_axis = eye - center
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[0, :3] = x_axis
        view_matrix[1, :3] = y_axis
        view_matrix[2, :3] = z_axis
        view_matrix[3, :3] = -np.array([np.dot(x_axis, eye), np.dot(y_axis, eye), np.dot(z_axis, eye)])
        return view_matrix.T
    
    def process_keyboard(self, direction, delta_time):
        velocity = CAMERA_SPEED * delta_time
        if direction == "FORWARD":
            self.position += self.front * velocity
        if direction == "BACKWARD":
            self.position -= self.front * velocity
        if direction == "LEFT":
            self.position -= self.right * velocity
        if direction == "RIGHT":
            self.position += self.right * velocity
        if direction == "UP":
            self.position += np.array([0.0, 1.0, 0.0], dtype=np.float32) * velocity
        if direction == "DOWN":
            self.position -= np.array([0.0, 1.0, 0.0], dtype=np.float32) * velocity


class VoxelShader:
    def __init__(self, ctx):
        self.ctx = ctx
        
        self.vertex_shader = """
        #version 330 core
        
        layout(location = 0) in vec3 in_position;
        layout(location = 1) in vec3 in_normal;
        layout(location = 2) in float in_face_id;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 frag_position;
        out vec3 frag_normal;
        out float v_face_id;
        
        void main() {
            frag_position = vec3(model * vec4(in_position, 1.0));
            frag_normal = mat3(transpose(inverse(model))) * in_normal;
            v_face_id = in_face_id;
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }
        """
        
        self.fragment_shader = """
        #version 330 core
        
        in vec3 frag_position;
        in vec3 frag_normal;
        in float v_face_id;
        
        uniform vec3 light_dir;
        uniform vec3 camera_pos;
        
        out vec4 frag_color;
        
        vec3 get_block_color(float face_id) {
            // Different colors for different block types (encoded in face_id high bits)
            uint type_id = uint(face_id * 16.0);
            if (type_id == 0u) return vec3(0.5, 0.35, 0.2);  // Dirt
            if (type_id == 1u) return vec3(0.4, 0.4, 0.4);   // Stone
            if (type_id == 2u) return vec3(0.2, 0.7, 0.2);   // Grass
            if (type_id == 3u) return vec3(0.6, 0.6, 0.1);   // Sand
            if (type_id == 4u) return vec3(0.3, 0.5, 0.8);   // Water (transparent)
            return vec3(0.8, 0.8, 0.8);  // Default
        }
        
        void main() {
            vec3 color = get_block_color(v_face_id);
            
            // Ambient lighting
            float ambient_strength = 0.3;
            vec3 ambient = ambient_strength * color;
            
            // Diffuse lighting
            vec3 norm = normalize(frag_normal);
            vec3 light_direction = normalize(light_dir);
            float diff = max(dot(norm, light_direction), 0.0);
            vec3 diffuse = diff * color;
            
            // Simple shading based on face direction
            vec3 final_color = ambient + diffuse;
            
            frag_color = vec4(final_color, 1.0);
        }
        """
        
        self.program = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )
        
        # Compute shader for terrain generation
        self.compute_shader = """
        #version 430 core
        
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
        
        layout(std430, binding = 0) buffer VoxelData {
            uint voxels[];
        };
        
        uniform int chunk_size;
        uniform int world_offset_x;
        uniform int world_offset_y;
        uniform int world_offset_z;
        uniform float seed;
        
        // Simple hash function
        uint hash(uint x) {
            x += (x << 10u);
            x ^= (x >> 6u);
            x += (x << 3u);
            x ^= (x >> 11u);
            x += (x << 15u);
            return x;
        }
        
        // Value noise
        float value_noise(vec3 p) {
            vec3 i = floor(p);
            vec3 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            
            uint n0 = hash(uint(i.x) + uint(i.y) * 57u + uint(i.z) * 113u);
            uint n1 = hash(uint(i.x + 1.0) + uint(i.y) * 57u + uint(i.z) * 113u);
            uint n2 = hash(uint(i.x) + uint(i.y + 1.0) * 57u + uint(i.z) * 113u);
            uint n3 = hash(uint(i.x + 1.0) + uint(i.y + 1.0) * 57u + uint(i.z) * 113u);
            uint n4 = hash(uint(i.x) + uint(i.y) * 57u + uint(i.z + 1.0) * 113u);
            uint n5 = hash(uint(i.x + 1.0) + uint(i.y) * 57u + uint(i.z + 1.0) * 113u);
            uint n6 = hash(uint(i.x) + uint(i.y + 1.0) * 57u + uint(i.z + 1.0) * 113u);
            uint n7 = hash(uint(i.x + 1.0) + uint(i.y + 1.0) * 57u + uint(i.z + 1.0) * 113u);
            
            float v0 = float(n0) / float(0xFFFFFFFFu);
            float v1 = float(n1) / float(0xFFFFFFFFu);
            float v2 = float(n2) / float(0xFFFFFFFFu);
            float v3 = float(n3) / float(0xFFFFFFFFu);
            float v4 = float(n4) / float(0xFFFFFFFFu);
            float v5 = float(n5) / float(0xFFFFFFFFu);
            float v6 = float(n6) / float(0xFFFFFFFFu);
            float v7 = float(n7) / float(0xFFFFFFFFu);
            
            float nx = mix(v0, v1, f.x);
            float ny = mix(v2, v3, f.x);
            float nz = mix(v4, v5, f.x);
            float nw = mix(v6, v7, f.x);
            
            float xy = mix(nx, ny, f.y);
            float zy = mix(nz, nw, f.y);
            
            return mix(xy, zy, f.z);
        }
        
        // Fractal Brownian Motion
        float fbm(vec3 p) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 1.0;
            
            for (int i = 0; i < 5; i++) {
                value += amplitude * value_noise(p * frequency);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            
            return value;
        }
        
        void main() {
            ivec3 pos = ivec3(gl_GlobalInvocationID);
            
            if (pos.x >= chunk_size || pos.y >= chunk_size || pos.z >= chunk_size) {
                return;
            }
            
            int global_x = world_offset_x * chunk_size + pos.x;
            int global_y = world_offset_y * chunk_size + pos.y;
            int global_z = world_offset_z * chunk_size + pos.z;
            
            // Generate terrain height using FBM
            vec3 noise_pos = vec3(global_x * 0.01, global_z * 0.01, seed);
            float height = fbm(noise_pos) * 20.0 + 10.0;
            
            uint block_type = 0u;
            
            if (float(global_y) < height - 4.0) {
                block_type = 1u;  // Stone
            } else if (float(global_y) < height) {
                block_type = 0u;  // Dirt
            } else if (float(global_y) == floor(height)) {
                if (height < 12.0) {
                    block_type = 3u;  // Sand (near water level)
                } else {
                    block_type = 2u;  // Grass
                }
            } else if (float(global_y) < 10.0 && float(global_y) >= height) {
                block_type = 4u;  // Water
            }
            
            int index = pos.z * chunk_size * chunk_size + pos.y * chunk_size + pos.x;
            voxels[index] = block_type;
        }
        """


class Chunk:
    def __init__(self, ctx, shader_program, position=(0, 0, 0)):
        self.ctx = ctx
        self.shader_program = shader_program
        self.position = position
        self.voxels = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint32)
        self.vertices = None
        self.vao = None
        self.vertex_buffer = None
        self.is_dirty = True
        
    def generate_mesh(self):
        """Generate mesh from voxel data using greedy meshing or simple culling"""
        vertices = []
        
        for x in range(CHUNK_SIZE):
            for y in range(CHUNK_SIZE):
                for z in range(CHUNK_SIZE):
                    block_type = self.voxels[x, y, z]
                    if block_type == 0:
                        continue
                    
                    # Check if block is visible (at least one face exposed)
                    visible = False
                    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if nx < 0 or nx >= CHUNK_SIZE or ny < 0 or ny >= CHUNK_SIZE or nz < 0 or nz >= CHUNK_SIZE:
                            visible = True
                            break
                        if self.voxels[nx, ny, nz] == 0:
                            visible = True
                            break
                    
                    if not visible:
                        continue
                    
                    # Add cube faces
                    self.add_cube_faces(vertices, x, y, z, block_type)
        
        if len(vertices) == 0:
            self.vao = None
            return
        
        vertex_data = np.array(vertices, dtype=np.float32)
        self.vertex_buffer = self.ctx.buffer(vertex_data.tobytes())
        
        # 3 floats for position, 3 for normal, 1 for face_id/block_type
        self.vao = self.ctx.simple_vertex_array(
            self.shader_program,
            self.vertex_buffer,
            'in_position', 'in_normal', 'in_face_id',
            stride=7 * 4
        )
        
    def add_cube_faces(self, vertices, x, y, z, block_type):
        """Add visible faces of a cube to the vertex list"""
        # Define cube vertices and normals for each face
        faces = [
            # Right face (x+)
            {'normal': (1, 0, 0), 'verts': [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]},
            # Left face (x-)
            {'normal': (-1, 0, 0), 'verts': [(0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1)]},
            # Top face (y+)
            {'normal': (0, 1, 0), 'verts': [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)]},
            # Bottom face (y-)
            {'normal': (0, -1, 0), 'verts': [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]},
            # Front face (z+)
            {'normal': (0, 0, 1), 'verts': [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]},
            # Back face (z-)
            {'normal': (0, 0, -1), 'verts': [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]},
        ]
        
        for face in faces:
            # Check if this face is visible
            face_visible = False
            nx, ny, nz = x, y, z
            if face['normal'][0] > 0: nx += 1
            elif face['normal'][0] < 0: nx -= 1
            if face['normal'][1] > 0: ny += 1
            elif face['normal'][1] < 0: ny -= 1
            if face['normal'][2] > 0: nz += 1
            elif face['normal'][2] < 0: nz -= 1
            
            if nx < 0 or nx >= CHUNK_SIZE or ny < 0 or ny >= CHUNK_SIZE or nz < 0 or nz >= CHUNK_SIZE:
                face_visible = True
            elif self.voxels[nx, ny, nz] == 0:
                face_visible = True
            
            if not face_visible:
                continue
            
            # Add two triangles for this face
            verts = face['verts']
            normal = face['normal']
            face_id = float(block_type) / 16.0
            
            # Triangle 1
            vertices.extend([x + verts[0][0], y + verts[0][1], z + verts[0][2]])
            vertices.extend(normal)
            vertices.append(face_id)
            
            vertices.extend([x + verts[1][0], y + verts[1][1], z + verts[1][2]])
            vertices.extend(normal)
            vertices.append(face_id)
            
            vertices.extend([x + verts[2][0], y + verts[2][1], z + verts[2][2]])
            vertices.extend(normal)
            vertices.append(face_id)
            
            # Triangle 2
            vertices.extend([x + verts[0][0], y + verts[0][1], z + verts[0][2]])
            vertices.extend(normal)
            vertices.append(face_id)
            
            vertices.extend([x + verts[2][0], y + verts[2][1], z + verts[2][2]])
            vertices.extend(normal)
            vertices.append(face_id)
            
            vertices.extend([x + verts[3][0], y + verts[3][1], z + verts[3][2]])
            vertices.extend(normal)
            vertices.append(face_id)


class World:
    def __init__(self, ctx, shader_program):
        self.ctx = ctx
        self.shader_program = shader_program
        self.chunks = {}
        self.compute_program = None
        self.voxel_buffer = None
        self.setup_compute_shader()
        
    def setup_compute_shader(self):
        self.compute_program = self.ctx.compute_shader(VoxelShader(self.ctx).compute_shader)
        
    def get_chunk_key(self, x, y, z):
        return (x, y, z)
    
    def get_or_create_chunk(self, x, y, z):
        key = self.get_chunk_key(x, y, z)
        if key not in self.chunks:
            chunk = Chunk(self.ctx, self.shader_program, (x, y, z))
            self.generate_chunk_gpu(chunk)
            self.chunks[key] = chunk
        return self.chunks[key]
    
    def generate_chunk_gpu(self, chunk):
        """Use compute shader to generate voxel data on GPU"""
        # Create voxel buffer
        voxel_count = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
        voxel_data = np.zeros(voxel_count, dtype=np.uint32)
        self.voxel_buffer = self.ctx.buffer(voxel_data.tobytes())
        
        # Bind buffer to compute shader
        self.compute_program['VoxelData'] = self.voxel_buffer
        self.compute_program['chunk_size'] = CHUNK_SIZE
        self.compute_program['world_offset_x'] = chunk.position[0]
        self.compute_program['world_offset_y'] = chunk.position[1]
        self.compute_program['world_offset_z'] = chunk.position[2]
        self.compute_program['seed'] = 42.0
        
        # Run compute shader
        grid_size = ((CHUNK_SIZE + 7) // 8, (CHUNK_SIZE + 7) // 8, (CHUNK_SIZE + 7) // 8)
        self.compute_program.run(grid_size=grid_size)
        
        # Read back voxel data
        voxel_data = np.frombuffer(self.voxel_buffer.read(), dtype=np.uint32).reshape((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE))
        chunk.voxels = voxel_data
        chunk.is_dirty = True
        
    def update(self, camera_position):
        """Update chunks based on camera position"""
        chunk_x = int(camera_position[0] / CHUNK_SIZE)
        chunk_y = int(camera_position[1] / CHUNK_SIZE)
        chunk_z = int(camera_position[2] / CHUNK_SIZE)
        
        # Load chunks around player
        for x in range(chunk_x - RENDER_DISTANCE, chunk_x + RENDER_DISTANCE + 1):
            for y in range(chunk_y - RENDER_DISTANCE, chunk_y + RENDER_DISTANCE + 1):
                for z in range(chunk_z - RENDER_DISTANCE, chunk_z + RENDER_DISTANCE + 1):
                    self.get_or_create_chunk(x, y, z)
        
        # Remove far chunks
        keys_to_remove = []
        for key in self.chunks:
            dx = abs(key[0] - chunk_x)
            dy = abs(key[1] - chunk_y)
            dz = abs(key[2] - chunk_z)
            if dx > RENDER_DISTANCE + 1 or dy > RENDER_DISTANCE + 1 or dz > RENDER_DISTANCE + 1:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.chunks[key]
        
        # Regenerate meshes for dirty chunks
        for chunk in self.chunks.values():
            if chunk.is_dirty:
                chunk.generate_mesh()
                chunk.is_dirty = False
    
    def render(self, view_matrix, projection_matrix, camera_pos):
        """Render all chunks"""
        for chunk in self.chunks.values():
            if chunk.vao is None:
                continue
            
            model_matrix = np.eye(4, dtype=np.float32)
            model_matrix[3, 0] = chunk.position[0] * CHUNK_SIZE
            model_matrix[3, 1] = chunk.position[1] * CHUNK_SIZE
            model_matrix[3, 2] = chunk.position[2] * CHUNK_SIZE
            
            self.shader_program['model'].write(model_matrix.tobytes())
            self.shader_program['view'].write(view_matrix.tobytes())
            self.shader_program['projection'].write(projection_matrix.tobytes())
            self.shader_program['light_dir'].write(np.array([0.5, 1.0, 0.3], dtype=np.float32).tobytes())
            self.shader_program['camera_pos'].write(np.array(camera_pos, dtype=np.float32).tobytes())
            
            chunk.vao.render(moderngl.TRIANGLES)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Space Engineers Clone - GPU Driven Procedural World")
        
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear_color = (0.5, 0.7, 1.0)
        
        self.camera = Camera()
        self.shader = VoxelShader(self.ctx)
        self.world = World(self.ctx, self.shader.program)
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.mouse_captured = False
        
        # Projection matrix
        self.projection_matrix = self.perspective(FOV, WINDOW_WIDTH / WINDOW_HEIGHT, NEAR, FAR)
        
        # Capture mouse for FPS controls
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.mouse_captured = True
        
    def perspective(self, fov, aspect, near, far):
        tan_half_fov = math.tan(math.radians(fov / 2))
        projection_matrix = np.zeros((4, 4), dtype=np.float32)
        projection_matrix[0, 0] = 1.0 / (aspect * tan_half_fov)
        projection_matrix[1, 1] = 1.0 / tan_half_fov
        projection_matrix[2, 2] = -(far + near) / (far - near)
        projection_matrix[2, 3] = -(2.0 * far * near) / (far - near)
        projection_matrix[3, 2] = -1.0
        return projection_matrix
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    if self.mouse_captured:
                        pygame.event.set_grab(False)
                        pygame.mouse.set_visible(True)
                        self.mouse_captured = False
                    else:
                        pygame.event.set_grab(True)
                        pygame.mouse.set_visible(False)
                        self.mouse_captured = True
    
    def handle_input(self, delta_time):
        if not self.mouse_captured:
            return
        
        keys = pygame.key.get_pressed()
        if keys[K_w] or keys[K_UP]:
            self.camera.process_keyboard("FORWARD", delta_time)
        if keys[K_s] or keys[K_DOWN]:
            self.camera.process_keyboard("BACKWARD", delta_time)
        if keys[K_a]:
            self.camera.process_keyboard("LEFT", delta_time)
        if keys[K_d]:
            self.camera.process_keyboard("RIGHT", delta_time)
        if keys[K_SPACE]:
            self.camera.process_keyboard("UP", delta_time)
        if keys[K_LSHIFT] or keys[K_RSHIFT]:
            self.camera.process_keyboard("DOWN", delta_time)
        
        # Mouse look
        if self.mouse_captured:
            mouse_delta = pygame.mouse.get_rel()
            self.camera.yaw += mouse_delta[0] * MOUSE_SENSITIVITY * 100
            self.camera.pitch -= mouse_delta[1] * MOUSE_SENSITIVITY * 100
            self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
            self.camera.update_direction()
    
    def run(self):
        while self.running:
            delta_time = self.clock.tick(60) / 1000.0
            
            self.handle_events()
            self.handle_input(delta_time)
            
            # Update world
            self.world.update(self.camera.position)
            
            # Render
            self.ctx.clear(color=True, depth=True)
            
            view_matrix = self.camera.get_view_matrix()
            self.world.render(view_matrix, self.projection_matrix, self.camera.position)
            
            pygame.display.flip()
        
        pygame.quit()


if __name__ == "__main__":
    print("Starting Space Engineers-like GPU-driven procedural world...")
    print("Controls:")
    print("  WASD - Move")
    print("  Space/Shift - Up/Down")
    print("  Mouse - Look around")
    print("  ESC - Toggle mouse capture")
    game = Game()
    game.run()
