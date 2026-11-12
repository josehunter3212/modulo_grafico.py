# modulo_grafico.pyimport numpy as np
from PIL import Image


# ==============================================================================
# 1. CLASE TEXTURE (Texturizado y Filtro Bilineal)
# ==============================================================================
class Texture:

    def __init__(self, width=128, height=128, image_path=None):
        if image_path:
            img = Image.open(image_path).convert("RGBA")
            self.data = np.array(img, dtype=np.float32) / 255.0
            self.height, self.width, _ = self.data.shape
        else:
            # Genera una textura procedural (tablero de ajedrez colorido)
            self.width = width
            self.height = height
            self.data = np.zeros((height, width, 4), dtype=np.float32)
            for y in range(height):
                for x in range(width):
                    is_even = ((x // 16) + (y // 16)) % 2 == 0
                    if is_even:
                        self.data[y, x] = [0.2, 0.6, 0.9, 1.0]  # Azul claro
                    else:
                        self.data[y, x] = [0.95, 0.95, 0.95, 1.0]  # Blanco

    def sample_bilinear(self, u, v):
        """Muestreo de textura usando interpolación bilineal."""
        # Repetición de coordenadas UV (Wrap)
        u = u % 1.0
        v = (1.0 - v) % 1.0  # Invertir V para coordenadas de imagen

        x = u * (self.width - 1)
        y = v * (self.height - 1)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)

        dx = x - x0
        dy = y - y0

        # Obtener los 4 texels vecinos
        c00 = self.data[y0, x0]
        c10 = self.data[y0, x1]
        c01 = self.data[y1, x0]
        c11 = self.data[y1, x1]

        # Interpolación horizontal y luego vertical
        c0 = c00 * (1.0 - dx) + c10 * dx
        c1 = c01 * (1.0 - dx) + c11 * dx
        return c0 * (1.0 - dy) + c1 * dy


# ==============================================================================
# 2. CLASE LIGHT (Modelo de Iluminación Phong)
# ==============================================================================
class Light:

    def __init__(self, position, color=(1.0, 1.0, 1.0), intensity=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32) * intensity
        self.ambient_strength = 0.2
        self.specular_strength = 0.6
        self.shininess = 32


# ==============================================================================
# 3. CLASE CAMERA (Transformaciones View y Projection)
# ==============================================================================
class Camera:

    def __init__(self, position, target, up=(0, 1, 0), fov=60.0, aspect=1.0, near=0.1, far=100.0):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def get_view_matrix(self):
        """Construye la matriz LookAt (Mundo -> Vista)."""
        z_axis = self.position - self.target
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(self.up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        view = np.identity(4, dtype=np.float32)
        view[0, :3] = x_axis
        view[1, :3] = y_axis
        view[2, :3] = z_axis
        view[0, 3] = -np.dot(x_axis, self.position)
        view[1, 3] = -np.dot(y_axis, self.position)
        view[2, 3] = -np.dot(z_axis, self.position)
        return view

    def get_projection_matrix(self):
        """Construye la matriz de Proyección en Perspectiva."""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        return proj


# ==============================================================================
# 4. CLASE MESH (Geometría y Transformaciones de Modelo)
# ==============================================================================
class Mesh:

    def __init__(self, vertices, normals, uvs, indices, texture=None, color=(1.0, 1.0, 1.0), alpha=1.0):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.uvs = np.array(uvs, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)
        self.texture = texture
        self.color = np.array(color, dtype=np.float32)
        self.alpha = alpha

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Grados
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def get_model_matrix(self):
        """Construye la matriz de Modelo (Local -> Mundo): T * R * S."""
        # Escala
        S = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0]).astype(np.float32)

        # Rotación (Euler XYZ)
        rx, ry, rz = np.radians(self.rotation)
        Rx = np.array([,
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
,
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
,
            [0, 0, 0, 1]
        ], dtype=np.float32)
        R = Rz @ Ry @ Rx

        # Traslación
        T = np.identity(4, dtype=np.float32)
        T[:3, 3] = self.position

        return T @ R @ S

    @staticmethod
    def create_cube(texture=None, color=(1.0, 1.0, 1.0), alpha=1.0):
        """Crea un cubo 3D unitario con normales y coordenadas UV por cara."""
        # 24 vértices (4 por cara para mantener normales planas y UVs independientes)
        v = [
            # Frente (+Z)
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            # Atrás (-Z)
            [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
            # Izquierda (-X)
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
            # Derecha (+X)
            [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
            # Arriba (+Y)
            [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
            # Abajo (-Y)
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
        ]
        n = [] * 4 + [[0, 0, -1]] * 4 + [[-1, 0, 0]] * 4 + [] * 4 + [[0, 1, 0]] * 4 + [[0, -1, 0]] * 4
        uv = [[0, 0], [1, 0], [1, 1], [0, 1]] * 6
        idx = []
        for i in range(6):
            base = i * 4
            idx.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
        return Mesh(v, n, uv, idx, texture, color, alpha)

    @staticmethod
    def create_quad(texture=None, color=(1.0, 1.0, 1.0), alpha=1.0):
        """Crea un plano cuadrado 2D en el espacio 3D."""
        v = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
        n = [[0, 0, 1]] * 4
        uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
        idx = [[0, 1, 2], [0, 2, 3]]
        return Mesh(v, n, uv, idx, texture, color, alpha)


# ==============================================================================
# 5. CLASE SCENE (Contenedor de Objetos, Luces y Cámara)
# ==============================================================================
class Scene:

    def __init__(self, camera):
        self.camera = camera
        self.meshes = []
        self.lights = []

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def add_light(self, light):
        self.lights.append(light)


# ==============================================================================
# 6. CLASE RENDERER (Pipeline, Gouraud/Phong, Z-Buffer, Alpha y Post-Proceso)
# ==============================================================================
class Renderer:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.color_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self.z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    def clear(self, bg_color=(0.1, 0.12, 0.15)):
        self.color_buffer[:] = bg_color
        self.z_buffer[:] = np.inf

    def _compute_phong_vertex(self, world_pos, world_normal, base_color, scene):
        """Calcula la iluminación de Phong por vértice (para sombreado Gouraud)."""
        N = world_normal / (np.linalg.norm(world_normal) + 1e-8)
        V = scene.camera.position - world_pos
        V = V / (np.linalg.norm(V) + 1e-8)

        final_light = np.zeros(3, dtype=np.float32)

        for light in scene.lights:
            # Componente Ambiental
            ambient = light.ambient_strength * light.color

            # Componente Difusa
            L = light.position - world_pos
            L = L / (np.linalg.norm(L) + 1e-8)
            diff = max(np.dot(N, L), 0.0)
            diffuse = diff * light.color

            # Componente Especular
            R = 2.0 * np.dot(N, L) * N - L
            R = R / (np.linalg.norm(R) + 1e-8)
            spec = (max(np.dot(V, R), 0.0)) ** light.shininess
            specular = light.specular_strength * spec * light.color

final_light += (ambient + diffuse + specular)
return np.clip(base_color * final_light, 0.0, 1.0)
def render(self, scene):
self.clear()
view_mat = scene.camera.get_view_matrix()
proj_mat = scene.camera.get_projection_matrix()
# Separar mallas opacas y transparentes (Painter's Algorithm para transparencia)
opaque_meshes = [m for m in scene.meshes if m.alpha >= 1.0]
transparent_meshes = [m for m in scene.meshes if m.alpha < 1.0]
# Ordenar transparentes de atrás hacia adelante según distancia a la cámara
transparent_meshes.sort(
key=lambda m: np.linalg.norm(scene.camera.position - m.position),
reverse=True
)
# Renderizar primero opacos, luego transparentes
for mesh in opaque_meshes + transparent_meshes:
self._render_mesh(mesh, view_mat, proj_mat, scene)
def _render_mesh(self, mesh, view_mat, proj_mat, scene):
model_mat = mesh.get_model_matrix()
normal_mat = np.linalg.inv(model_mat[:3, :3]).T
mvp = proj_mat @ view_mat @ model_mat
# 1. Vertex Shader (Transformaciones + Iluminación Gouraud)
num_verts = len(mesh.vertices)
screen_coords = np.zeros((num_verts, 3), dtype=np.float32)
inv_w = np.zeros(num_verts, dtype=np.float32)
vertex_colors = np.zeros((num_verts, 3), dtype=np.float32)
for i in range(num_verts):
v_local = np.append(mesh.vertices[i], 1.0)
v_world = (model_mat @ v_local)[:3]
n_world = normal_mat @ mesh.normals[i]
# Iluminación Phong evaluada en el vértice (Sombreado Gouraud)
vertex_colors[i] = self._compute_phong_vertex(v_world, n_world, mesh.color, scene)
# Proyección a Clip Space
v_clip = mvp @ v_local
w = v_clip[3] if abs(v_clip[3]) > 1e-6 else 1e-6
inv_w[i] = 1.0 / w
# División Perspectiva -> NDC [-1, 1]
ndc = v_clip[:3] * inv_w[i]
# Transformación de Viewport -> Píxeles de pantalla
sx = (ndc[0] + 1.0) * 0.5 * (self.width - 1)
sy = (1.0 - ndc[1]) * 0.5 * (self.height - 1)  # Y invertida en pantalla
sz = ndc[2]
screen_coords[i] = [sx, sy, sz]
# 2. Rasterización de triángulos
for tri in mesh.indices:
i0, i1, i2 = tri
self._rasterize_triangle(
screen_coords[i0], screen_coords[i1], screen_coords[i2],
vertex_colors[i0], vertex_colors[i1], vertex_colors[i2],
mesh.uvs[i0], mesh.uvs[i1], mesh.uvs[i2],
inv_w[i0], inv_w[i1], inv_w[i2],
mesh.texture, mesh.alpha
)
def _rasterize_triangle(self, p0, p1, p2, c0, c1, c2, uv0, uv1, uv2, iw0, iw1, iw2, texture, alpha):
# Back-face culling básico en espacio de pantalla
area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
if area >= 0:  # Triángulo mirando hacia atrás (CCW en pantalla con Y invertida)
return
# Bounding box del triángulo recortado a los límites de pantalla
min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
max_x = min(self.width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
max_y = min(self.height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
for y in range(min_y, max_y + 1):
for x in range(min_x, max_x + 1):
px, py = x + 0.5, y + 0.5
# Coordenadas baricéntricas
w0 = ((p1[0] - px) * (p2[1] - py) - (p1[1] - py) * (p2[0] - px)) / area
w1 = ((p2[0] - px) * (p0[1] - py) - (p2[1] - py) * (p0[0] - px)) / area
w2 = 1.0 - w0 - w1
if w0 >= 0 and w1 >= 0 and w2 >= 0:
# Interpolación de profundidad Z
z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
if z < self.z_buffer[y, x]:
# Interpolación Gouraud del color iluminado
shaded_color = w0 * c0 + w1 * c1 + w2 * c2
final_alpha = alpha
# Texturizado con corrección de perspectiva y filtro bilineal
if texture is not None:
interp_iw = w0 * iw0 + w1 * iw1 + w2 * iw2
u = (w0 * uv0[0] * iw0 + w1 * uv1[0] * iw1 + w2 * uv2[0] * iw2) / interp_iw
v = (w0 * uv0[1] * iw0 + w1 * uv1[1] * iw1 + w2 * uv2[1] * iw2) / interp_iw
texel = texture.sample_bilinear(u, v)
shaded_color = shaded_color * texel[:3]
final_alpha *= texel[3]
# Alpha Blending vs Escritura Opaca
if final_alpha >= 1.0:
self.z_buffer[y, x] = z
self.color_buffer[y, x] = shaded_color
else:
# Mezcla alfa sin sobrescribir el Z-buffer
bg = self.color_buffer[y, x]
self.color_buffer[y, x] = final_alpha * shaded_color + (1.0 - final_alpha) * bg
def apply_post_processing(self, effect="vignette_and_sharpen"):
"""Aplica efectos 2D sobre el framebuffer final."""
if effect == "grayscale":
lum = (0.299 * self.color_buffer[:, :, 0] +
0.587 * self.color_buffer[:, :, 1] +
0.114 * self.color_buffer[:, :, 2])
self.color_buffer = np.stack([lum, lum, lum], axis=-1)
elif effect == "box_blur":
padded = np.pad(self.color_buffer, ((1, 1), (1, 1), (0, 0)), mode='edge')
blurred = (
padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
) / 9.0
self.color_buffer = blurred
elif effect == "vignette_and_sharpen":
# 1. Filtro de enfoque (Sharpen kernel 3x3)
padded = np.pad(self.color_buffer, ((1, 1), (1, 1), (0, 0)), mode='edge')
sharpened = (
5.0 * padded[1:-1, 1:-1] -
padded[:-2, 1:-1] - padded[2:, 1:-1] -
padded[1:-1, :-2] - padded[1:-1, 2:]
)
sharpened = np.clip(sharpened, 0.0, 1.0)
# 2. Efecto Viñeta (oscurecimiento suave hacia los bordes)
y, x = np.ogrid[:self.height, :self.width]
cx, cy = self.width / 2.0, self.height / 2.0
dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
vignette = np.clip(1.0 - 0.35 * (dist ** 2), 0.0, 1.0)[:, :, np.newaxis]
self.color_buffer = sharpened * vignette
def save_image(self, filename="render_output.png"):
img_uint8 = (np.clip(self.color_buffer, 0.0, 1.0) * 255).astype(np.uint8)
Image.fromarray(img_uint8).save(filename)
print(f"Imagen guardada exitosamente en: {filename}")

==============================================================================


7. EJECUCIÓN DE PRUEBA (Escena 3D Completa)


==============================================================================

if name == "main":
WIDTH, HEIGHT = 600, 600
# 1. Configurar cámara y escena
camera = Camera(position=(0.0, 1.2, 3.2), target=(0.0, 0.0, 0.0), aspect=WIDTH / HEIGHT)
scene = Scene(camera)
# 2. Agregar luz puntual
light = Light(position=(2.5, 3.0, 2.5), color=(1.0, 0.98, 0.95), intensity=1.2)
scene.add_light(light)
# 3. Crear cubo opaco con textura procedural (filtro bilineal)
checker_tex = Texture(width=128, height=128)
cube = Mesh.create_cube(texture=checker_tex, color=(1.0, 1.0, 1.0), alpha=1.0)
cube.rotation = np.array([20.0, 35.0, 0.0])
cube.position = np.array([-0.2, 0.0, 0.0])
scene.add_mesh(cube)
# 4. Crear panel semitransparente frente al cubo (Alpha Blending)
glass_panel = Mesh.create_quad(texture=None, color=(1.0, 0.25, 0.35), alpha=0.45)
glass_panel.position = np.array([0.35, -0.1, 1.1])
glass_panel.scale = np.array([1.3, 1.3, 1.0])
glass_panel.rotation = np.array([0.0, -15.0, 0.0])
scene.add_mesh(glass_panel)
# 5. Renderizar y aplicar post-procesado
renderer = Renderer(WIDTH, HEIGHT)
renderer.render(scene)
renderer.apply_post_processing(effect="vignette_and_sharpen")
renderer.save_image("render_output.png")import numpy as np


def translation_matrix(tx, ty, tz):
    mat = np.identity(4, dtype=np.float32)
    mat[:3, 3] = [tx, ty, tz]
    return mat


def scale_matrix(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0]).astype(np.float32)


def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    Rx = np.array(
        [,
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
,
        ],
        dtype=np.float32,
    )

    Ry = np.array(
        [
            [np.cos(ry), 0, np.sin(ry), 0],
,
            [-np.sin(ry), 0, np.cos(ry), 0],
,
        ],
        dtype=np.float32,
    )

    Rz = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
,
 ,
        ],
        dtype=np.float32,
    )

    return Rz @ Ry @ Rx


def look_at_matrix(eye, target, up):
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = x_axis
    view[1, :3] = y_axis
    view[2, :3] = z_axis
    view[0, 3] = -np.dot(x_axis, eye)
    view[1, 3] = -np.dot(y_axis, eye)
    view[2, 3] = -np.dot(z_axis, eye)
    return view


def perspective_matrix(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj
import numpy as np
from PIL import Image


class Texture:

    def __init__(self, width=128, height=128, image_path=None):
        if image_path:
            img = Image.open(image_path).convert("RGBA")
            self.data = np.array(img, dtype=np.float32) / 255.0
            self.height, self.width, _ = self.data.shape
        else:
            self.width = width
            self.height = height
            self.data = self._generate_checkerboard(width, height)

    def _generate_checkerboard(self, w, h):
        tex = np.zeros((h, w, 4), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                even = ((x // 16) + (y // 16)) % 2 == 0
                if even:
                    tex[y, x] = [0.15, 0.55, 0.85, 1.0]
                else:
                    tex[y, x] = [0.95, 0.95, 0.95, 1.0]
        return tex

    def sample_bilinear(self, u, v):
        """Muestreo bilineal vectorizado/escalar con repetición de bordes."""
        u = u % 1.0
        v = (1.0 - v) % 1.0

        x = u * (self.width - 1)
        y = v * (self.height - 1)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)

        dx = x - x0
        dy = y - y0

        c00 = self.data[y0, x0]
        c10 = self.data[y0, x1]
        c01 = self.data[y1, x0]
        c11 = self.data[y1, x1]

        top = c00 * (1.0 - dx) + c10 * dx
        bottom = c01 * (1.0 - dx) + c11 * dx
        return top * (1.0 - dy) + bottom * dy
from math_utils import rotation_matrix_xyz, scale_matrix, translation_matrix
import numpy as np


class Mesh:

    def __init__(
        self,
        vertices,
        normals,
        uvs,
        indices,
        texture=None,
        color=(1.0, 1.0, 1.0),
        alpha=1.0,
    ):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.uvs = np.array(uvs, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.int32)
        self.texture = texture
        self.color = np.array(color, dtype=np.float32)
        self.alpha = alpha

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def get_model_matrix(self):
        T = translation_matrix(*self.position)
        R = rotation_matrix_xyz(*self.rotation)
        S = scale_matrix(*self.scale)
        return T @ R @ S

    @staticmethod
    def load_obj(filepath, texture=None, color=(1.0, 1.0, 1.0), alpha=1.0):
        """Carga un modelo 3D en formato Wavefront .OBJ (triangulado)."""
        raw_v, raw_vt, raw_vn = [], [], []
        out_v, out_vt, out_vn, out_idx = [], [], [], []

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    raw_v.append([float(x) for x in parts[1:4]])
                elif parts[0] == "vt":
                    raw_vt.append([float(x) for x in parts[1:3]])
                elif parts[0] == "vn":
                    raw_vn.append([float(x) for x in parts[1:4]])
                elif parts[0] == "f":
                    face_verts = parts[1:]
                    # Triangulación tipo abanico (fan) para polígonos de N lados
                    for i in range(1, len(face_verts) - 1):
                        tri = [face_verts[0], face_verts[i], face_verts[i + 1]]
                        tri_indices = []
                        for item in tri:
                            vals = item.split("/")
                            vi = int(vals[0]) - 1
                            vti = (
                                int(vals[1]) - 1
                                if len(vals) > 1 and vals[1]
                                else -1
                            )
                            vni = (
                                int(vals[2]) - 1
                                if len(vals) > 2 and vals[2]
                                else -1
                            )

                            out_v.append(raw_v[vi])
                            out_vt.append(raw_vt[vti] if vti >= 0 else [0.0, 0.0])
                            out_vn.append(
                                raw_vn[vni] if vni >= 0 else [0.0, 1.0, 0.0]
                            )
                            tri_indices.append(len(out_v) - 1)
                        out_idx.append(tri_indices)

        return Mesh(out_v, out_vn, out_vt, out_idx, texture, color, alpha)

    @staticmethod
    def create_cube(texture=None, color=(1.0, 1.0, 1.0), alpha=1.0):
        v = [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],  # Frente
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],  # Atrás
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],  # Izquierda
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],  # Derecha
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],  # Arriba
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],  # Abajo
        ]
        n = (
            [[0, 0, 1]] * 4
            + [[0, 0, -1]] * 4
            + [[-1, 0, 0]] * 4
            + [[1, 0, 0]] * 4
            + [[0, 1, 0]] * 4
            + [[0, -1, 0]] * 4
        )
        uv = [[0, 0], [1, 0], [1, 1], [0, 1]] * 6
        idx = []
        for i in range(6):
            b = i * 4
            idx.extend([[b, b + 1, b + 2], [b, b + 2, b + 3]])
        return Mesh(v, n, uv, idx, texture, color, alpha)

    @staticmethod
    def create_quad(texture=None, color=(1.0, 1.0, 1.0), alpha=1.0):
        v = [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ]
        n = [[0, 0, 1]] * 4
        uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
        idx = [[0, 1, 2], [0, 2, 3]]
        return Mesh(v, n, uv, idx, texture, color, alpha)
from math_utils import look_at_matrix, perspective_matrix
import numpy as np


class Light:

    def __init__(self, position, color=(1.0, 1.0, 1.0), intensity=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32) * intensity
        self.ambient_strength = 0.2
        self.specular_strength = 0.7
        self.shininess = 32


class Camera:

    def __init__(
        self,
        position,
        target,
        up=(0, 1, 0),
        fov=60.0,
        aspect=1.0,
        near=0.1,
        far=100.0,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def get_view_matrix(self):
        return look_at_matrix(self.position, self.target, self.up)

    def get_projection_matrix(self):
        return perspective_matrix(self.fov, self.aspect, self.near, self.far)


class Scene:

    def __init__(self, camera):
        self.camera = camera
        self.meshes = []
        self.lights = []

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def add_light(self, light):
        self.lights.append(light)
import numpy as np


class Renderer:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.color_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self.z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    def clear(self, bg_color=(0.08, 0.10, 0.14)):
        self.color_buffer[:] = bg_color
        self.z_buffer[:] = np.inf

    def _compute_phong_vertex(self, world_pos, world_normal, base_color, scene):
        """Ecuación de iluminación de Phong evaluada en los vértices (Gouraud Shading)."""
        N = world_normal / (np.linalg.norm(world_normal) + 1e-8)
        V = scene.camera.position - world_pos
        V /= np.linalg.norm(V) + 1e-8

        accum_light = np.zeros(3, dtype=np.float32)

        for light in scene.lights:
            # 1. Ambiental
            ambient = light.ambient_strength * light.color

            # 2. Difusa
            L = light.position - world_pos
            L /= np.linalg.norm(L) + 1e-8
            diff = max(np.dot(N, L), 0.0)
            diffuse = diff * light.color

            # 3. Especular
            R = 2.0 * np.dot(N, L) * N - L
            R /= np.linalg.norm(R) + 1e-8
            spec = (max(np.dot(V, R), 0.0)) ** light.shininess
            specular = light.specular_strength * spec * light.color

            accum_light += ambient + diffuse + specular

        return np.clip(base_color * accum_light, 0.0, 1.0)

    def render(self, scene):
        self.clear()
        view_mat = scene.camera.get_view_matrix()
        proj_mat = scene.camera.get_projection_matrix()

        # Separar geometría opaca y transparente
        opaque = [m for m in scene.meshes if m.alpha >= 1.0]
        transparent = [m for m in scene.meshes if m.alpha < 1.0]

        # Painter's Algorithm: ordenar objetos transparentes de lejos a cerca
        transparent.sort(
            key=lambda m: np.linalg.norm(scene.camera.position - m.position),
            reverse=True,
        )

        for mesh in opaque + transparent:
            self._render_mesh(mesh, view_mat, proj_mat, scene)

    def _render_mesh(self, mesh, view_mat, proj_mat, scene):
        model_mat = mesh.get_model_matrix()
        normal_mat = np.linalg.inv(model_mat[:3, :3]).T
        mvp = proj_mat @ view_mat @ model_mat

        num_verts = len(mesh.vertices)
        screen_coords = np.zeros((num_verts, 3), dtype=np.float32)
        inv_w = np.zeros(num_verts, dtype=np.float32)
        vertex_colors = np.zeros((num_verts, 3), dtype=np.float32)

        # Etapa de Vértices (Transformación + Iluminación Gouraud)
        for i in range(num_verts):
            v_local = np.append(mesh.vertices[i], 1.0)
            v_world = (model_mat @ v_local)[:3]
            n_world = normal_mat @ mesh.normals[i]

            vertex_colors[i] = self._compute_phong_vertex(
                v_world, n_world, mesh.color, scene
            )

            v_clip = mvp @ v_local
            w = v_clip[3] if abs(v_clip[3]) > 1e-6 else 1e-6
            inv_w[i] = 1.0 / w

            ndc = v_clip[:3] * inv_w[i]
            sx = (ndc[0] + 1.0) * 0.5 * (self.width - 1)
            sy = (1.0 - ndc[1]) * 0.5 * (self.height - 1)
            sz = ndc[2]
            screen_coords[i] = [sx, sy, sz]

        # Etapa de Rasterización
        for i0, i1, i2 in mesh.indices:
            self._rasterize_triangle(
                screen_coords[i0],
                screen_coords[i1],
                screen_coords[i2],
                vertex_colors[i0],
                vertex_colors[i1],
                vertex_colors[i2],
                mesh.uvs[i0],
                mesh.uvs[i1],
                mesh.uvs[i2],
                inv_w[i0],
                inv_w[i1],
                inv_w[i2],
                mesh.texture,
                mesh.alpha,
            )

    def _rasterize_triangle(
        self,
        p0,
        p1,
        p2,
        c0,
        c1,
        c2,
        uv0,
        uv1,
        uv2,
        iw0,
        iw1,
        iw2,
        texture,
        alpha,
    ):
        # Back-face culling
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (
            p2[0] - p0[0]
        )
        if area >= 0:
            return

        min_x = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        max_x = min(self.width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        min_y = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        max_y = min(self.height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))

        if min_x > max_x or min_y > max_y:
            return

        # Rejilla vectorizada de coordenadas dentro del bounding box del triángulo
        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        w0 = (
            (p1[0] - grid_x) * (p2[1] - grid_y)
            - (p1[1] - grid_y) * (p2[0] - grid_x)
        ) / area
        w1 = (
            (p2[0] - grid_x) * (p0[1] - grid_y)
            - (p2[1] - grid_y) * (p0[0] - grid_x)
        ) / area
        w2 = 1.0 - w0 - w1

        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(mask):
            return

        # Interpolación de profundidad Z
        z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
        sub_z = self.z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        depth_mask = mask & (z < sub_z) & (z >= -1.0) & (z <= 1.0)

        if not np.any(depth_mask):
            return

        # Índices válidos dentro del sub-bloque
        vy, vx = np.where(depth_mask)
        bw0, bw1, bw2 = w0[vy, vx], w1[vy, vx], w2[vy, vx]

        # Interpolación Gouraud
        shaded = (
            bw0[:, None] * c0 + bw1[:, None] * c1 + bw2[:, None] * c2
        )
        final_alpha = np.full(len(vy), alpha, dtype=np.float32)

        # Texturizado con corrección de perspectiva y filtro bilineal
        if texture is not None:
            interp_iw = bw0 * iw0 + bw1 * iw1 + bw2 * iw2
            u = (
                bw0 * uv0[0] * iw0 + bw1 * uv1[0] * iw1 + bw2 * uv2[0] * iw2
            ) / interp_iw
            v = (
                bw0 * uv0[1] * iw0 + bw1 * uv1[1] * iw1 + bw2 * uv2[1] * iw2
            ) / interp_iw

            for idx in range(len(vy)):
                texel = texture.sample_bilinear(u[idx], v[idx])
                shaded[idx] *= texel[:3]
                final_alpha[idx] *= texel[3]

        # Escritura en Framebuffer (Opaco vs Alpha Blending)
        py, px = vy + min_y, vx + min_x
        if alpha >= 1.0 and np.all(final_alpha >= 1.0):
            self.z_buffer[py, px] = z[vy, vx]
            self.color_buffer[py, px] = shaded
        else:
            bg = self.color_buffer[py, px]
            a = final_alpha[:, None]
            self.color_buffer[py, px] = a * shaded + (1.0 - a) * bg

    def apply_post_processing(self, mode="none"):
        """Aplica efectos 2D en espacio de pantalla sobre el framebuffer."""
        if mode == "none":
            return

        elif mode == "grayscale":
            lum = (
                0.299 * self.color_buffer[:, :, 0]
                + 0.587 * self.color_buffer[:, :, 1]
                + 0.114 * self.color_buffer[:, :, 2]
            )
            self.color_buffer = np.stack([lum, lum, lum], axis=-1)

        elif mode == "invert":
            self.color_buffer = 1.0 - self.color_buffer

        elif mode == "blur":
            padded = np.pad(
                self.color_buffer, ((1, 1), (1, 1), (0, 0)), mode="edge"
            )
            self.color_buffer = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + padded[1:-1, 1:-1]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 9.0

        elif mode == "vignette_sharpen":
            padded = np.pad(
                self.color_buffer, ((1, 1), (1, 1), (0, 0)), mode="edge"
            )
            sharpened = np.clip(
                5.0 * padded[1:-1, 1:-1]
                - padded[:-2, 1:-1]
                - padded[2:, 1:-1]
                - padded[1:-1, :-2]
                - padded[1:-1, 2:],
                0.0,
                1.0,
            )
            y, x = np.ogrid[: self.height, : self.width]
            cx, cy = self.width / 2.0, self.height / 2.0
            dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
            vignette = np.clip(1.0 - 0.4 * (dist**2), 0.0, 1.0)[:, :, None]
            self.color_buffer = sharpened * vignette
from mesh import Mesh
import numpy as np
import pygame
from renderer import Renderer
from scene import Camera, Light, Scene
from texture import Texture


def main():
    WIDTH, HEIGHT = 480, 360
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        "Motor Gráfico 3D por Software | Teclas: [1-5] Post-Proceso, [WASD] Cámara, [Espacio] Pausa"
    )
    clock = pygame.time.Clock()

    # 1. Inicializar Cámara y Escena
    camera = Camera(
        position=(0.0, 1.0, 3.5), target=(0.0, 0.0, 0.0), aspect=WIDTH / HEIGHT
    )
    scene = Scene(camera)

    # 2. Luces (Luz principal cálida + Luz de relleno fría)
    scene.add_light(
        Light(position=(3.0, 3.0, 3.0), color=(1.0, 0.95, 0.9), intensity=1.1)
    )
    scene.add_light(
        Light(position=(-3.0, -1.0, 2.0), color=(0.3, 0.5, 1.0), intensity=0.5)
    )

    # 3. Objetos de la escena
    tex_checker = Texture(width=128, height=128)
    main_cube = Mesh.create_cube(
        texture=tex_checker, color=(1.0, 1.0, 1.0), alpha=1.0
    )
    main_cube.position = np.array([-0.3, 0.0, 0.0])
    scene.add_mesh(main_cube)

    # Cristal semitransparente frente al cubo
    glass_shield = Mesh.create_quad(
        texture=None, color=(1.0, 0.2, 0.4), alpha=0.45
    )
    glass_shield.position = np.array([0.4, -0.1, 1.2])
    glass_shield.scale = np.array([1.2, 1.2, 1.0])
    glass_shield.rotation = np.array([0.0, -20.0, 0.0])
    scene.add_mesh(glass_shield)

    # 4. Motor de Renderizado
    renderer = Renderer(WIDTH, HEIGHT)
    post_effects = ["none", "vignette_sharpen", "grayscale", "blur", "invert"]
    current_effect = 0
    paused = False
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        # Manejo de eventos y controles
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_1:
                    current_effect = 0
                elif event.key == pygame.K_2:
                    current_effect = 1
                elif event.key == pygame.K_3:
                    current_effect = 2
                elif event.key == pygame.K_4:
                    current_effect = 3
                elif event.key == pygame.K_5:
                    current_effect = 4

        # Movimiento continuo de cámara (WASD)
        keys = pygame.key.get_pressed()
        cam_speed = 2.5 * dt
        if keys[pygame.K_w]:
            camera.position[2] -= cam_speed
        if keys[pygame.K_s]:
            camera.position[2] += cam_speed
        if keys[pygame.K_a]:
            camera.position[0] -= cam_speed
        if keys[pygame.K_d]:
            camera.position[0] += cam_speed

        # Animación de la geometría
        if not paused:
            main_cube.rotation[0] += 35.0 * dt
            main_cube.rotation[1] += 55.0 * dt

        # Ejecutar Pipeline Gráfico
        renderer.render(scene)
        renderer.apply_post_processing(mode=post_effects[current_effect])

        # Convertir Framebuffer a superficie Pygame y mostrar
        frame = (np.clip(renderer.color_buffer, 0.0, 1.0) * 255).astype(
            np.uint8
        )
        # Pygame espera formato (width, height, 3), por lo que transponemos ejes X e Y
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
