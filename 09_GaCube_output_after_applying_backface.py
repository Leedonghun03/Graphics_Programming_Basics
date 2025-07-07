import pygame
import numpy as np
import math

pygame.init()

screenWidth = 800
screenHeight = 600

BACKGROUND_COLR = (105, 105, 105)
LINE_COLOR = (255,255,255)
RedColor = (255, 0, 0)
BlueColor = (0, 0, 255)
GreenColor = (0, 255, 0)
PurpleColor = (255, 0, 255)

screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Graphics Programming")
running = True

texture = pygame.image.load('ga.jpg')

class Vertex:
    def __init__(self, position=None, uv=None):
        self.position = position if position is not None else np.array([1, 1, 1, 1])
        self.uv = uv if uv is not None else np.array([1, 1])
    def set_position(self, position) :
        self.position = position
    def set_uv(self, uv) :
        self.uv = uv

# 메쉬 데이터 초기화
# 정점 데이터 구성
cube_size = 100
half_size = cube_size / 2

tripoint = np.array([
    [-1* half_size, -1* half_size, 1* half_size, 1],
    [ 1* half_size, -1* half_size, 1* half_size, 1],
    [ 1* half_size, 1* half_size, 1* half_size, 1],
    [-1* half_size, 1* half_size, 1* half_size, 1],
    [-1* half_size, -1* half_size, -1* half_size, 1],
    [ 1* half_size, -1* half_size, -1* half_size, 1],
    [ 1* half_size, 1* half_size, -1* half_size, 1],
    [-1* half_size, 1* half_size, -1* half_size, 1]
])

vertices = [
    # FRONT
    Vertex(np.array(tripoint[0][:3]), np.array([0, 1])),
    Vertex(np.array(tripoint[1][:3]), np.array([1, 1])),
    Vertex(np.array(tripoint[2][:3]), np.array([1, 0])),
    Vertex(np.array(tripoint[3][:3]), np.array([0, 0])),

    # RIGHT
    Vertex(np.array(tripoint[1][:3]), np.array([0, 1])),
    Vertex(np.array(tripoint[5][:3]), np.array([1, 1])),
    Vertex(np.array(tripoint[6][:3]), np.array([1, 0])),
    Vertex(np.array(tripoint[2][:3]), np.array([0, 0])),

    # TOP
    Vertex(np.array(tripoint[3][:3]), np.array([0, 1])),
    Vertex(np.array(tripoint[2][:3]), np.array([1, 1])),
    Vertex(np.array(tripoint[6][:3]), np.array([1, 0])),
    Vertex(np.array(tripoint[7][:3]), np.array([0, 0])),

    # BACK
    Vertex(np.array(tripoint[5][:3]), np.array([0, 1])),
    Vertex(np.array(tripoint[4][:3]), np.array([1, 1])),
    Vertex(np.array(tripoint[7][:3]), np.array([1, 0])),
    Vertex(np.array(tripoint[6][:3]), np.array([0, 0])),

    # LEFT
    Vertex(np.array(tripoint[4][:3]), np.array([0, 1])),
    Vertex(np.array(tripoint[0][:3]), np.array([1, 1])),
    Vertex(np.array(tripoint[3][:3]), np.array([1, 0])),
    Vertex(np.array(tripoint[7][:3]), np.array([0, 0])),

    # BOTTOM
    Vertex(np.array(tripoint[4][:3]), np.array([0, 1])),
    Vertex(np.array(tripoint[5][:3]), np.array([1, 1])),
    Vertex(np.array(tripoint[1][:3]), np.array([1, 0])),
    Vertex(np.array(tripoint[0][:3]), np.array([0, 0])),
]

# 삼각형을 이루는 정점 인덱스 구축
trifaces = [
    [0, 1, 2], [2, 3, 0],     # FRONT
    [4, 5, 6], [6, 7, 4],     # RIGHT
    [8, 9,10], [10,11,8],     # TOP
    [12,13,14], [14,15,12],   # BACK
    [16,17,18], [18,19,16],   # LEFT
    [20,21,22], [22,23,20]    # BOTTOM
]                
                
# model matrix 구성
def ModelMatrix(translate_pos, scale_value, rotation_value):
    # 이동 변환 행렬
    translate_matrix = np.array([
        [1, 0, 0, translate_pos[0]],
        [0, 1, 0, translate_pos[1]],
        [0, 0, 1, translate_pos[2]],
        [0, 0, 0, 1]
    ])

    # 크기 변환 행렬
    scale_matrix = np.array([
        [scale_value[0], 0, 0, 0],
        [0, scale_value[1], 0, 0],
        [0, 0, scale_value[2], 0],
        [0, 0, 0, 1]
    ])

    # 회전 변환 행렬
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_value[0]), -np.sin(rotation_value[0]), 0],
        [0, np.sin(rotation_value[0]), np.cos(rotation_value[0]), 0],
        [0, 0, 0, 1]
    ])

    rotation_y = np.array([
        [np.cos(rotation_value[1]), 0, np.sin(rotation_value[1]), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_value[1]), 0, np.cos(rotation_value[1]),0],
        [0, 0, 0, 1]
    ])

    rotation_z = np.array([
        [np.cos(rotation_value[2]), -np.sin(rotation_value[2]), 0, 0],
        [np.sin(rotation_value[2]), np.cos(rotation_value[2]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Rotaiton : Ry.Rx.Rz
    rotation_matrix = np.matmul(rotation_y,np.matmul(rotation_x,rotation_z))

    # model matrix : T.R.S
    return np.matmul(translate_matrix,np.matmul(rotation_matrix,scale_matrix))

# view matrix 구성
# 카메라 위치
cam_pos = np.array([150, 100, 300])

# 바라보는 위치
target_pos = np.array([0, 0, 0])

# 카메라의 위쪽 방향 기준 벡터
up = np.array([0, 1, 0])

# x,y,z축은 모두 정규화 : 벡터의 노름을 반환하는 np.linalg.norm 이용
view_z = (target_pos - cam_pos)
view_z = view_z / np.linalg.norm(view_z)
view_x = np.cross(up, view_z)
view_x = view_x / np.linalg.norm(view_x)
view_y = np.cross(view_z, view_x)
view_y = view_y/ np.linalg.norm(view_y)

cam_inv_model_matrix = np.array([
    [view_x[0], view_x[1], view_x[2], -np.dot(view_x, cam_pos)],
    [view_y[0], view_y[1], view_y[2], -np.dot(view_y, cam_pos)],
    [view_z[0], view_z[1], view_z[2], -np.dot(view_z, cam_pos)],
    [0, 0, 0, 1]
])
rotation_y_180 = np.array([
    [-1, 0, 0, 0],
    [ 0, 1, 0, 0],
    [ 0, 0, -1, 0],
    [ 0, 0, 0, 1]
])

# y축 180를 회전시켜서 x,y축이 수학적인 2차원 좌표계처럼 보이게 수정
view_matrix = np.matmul(rotation_y_180,cam_inv_model_matrix)

# clip Space : projection matrix
fov = math.radians(60)

k = screenWidth / screenHeight # aspect ratio
n = 0.1 # near clip plane
f = 1000 # far clip plane

d = 1 / np.tan(fov / 2) # focal length

projection_matrix = np.array([
    [d/k, 0, 0, 0],
    [0, d, 0, 0],
    [0, 0, (n + f) / (n - f), (2 * n * f) / ( n - f)],
    [0, 0, -1, 0]
])

# screen space : viewport transform
viewport_matrix = np.array([
    [screenWidth / 2, 0, 0, screenWidth / 2],
    [0, -screenHeight / 2, 0, screenHeight / 2],
    [0, 0, 0.5, 0.5],
    [0, 0, 0, 1]
])

def baryentric_coords_ext(triangle, point):
    vector_u = triangle[1] - triangle[0]
    vector_v = triangle[2] - triangle[0]
    vector_w = point - triangle[0]
    
    dot_uv = np.float64(vector_u.dot(vector_v))
    dot_vv = np.float64(vector_v.dot(vector_v))
    dot_uu = np.float64(vector_u.dot(vector_u))
    
    inv_denom = 1 / (dot_uv * dot_uv - dot_vv * dot_uu)
    
    dot_wu = np.float64(vector_w.dot(vector_u))
    dot_wv = np.float64(vector_w.dot(vector_v))
    
    lambda1 = (dot_wv * dot_uv - dot_wu * dot_vv) * inv_denom
    lambda2 = (dot_wu * dot_uv - dot_wv * dot_uu) * inv_denom
    lambda3 = 1.0 - lambda1 - lambda2
    return (lambda3, lambda1, lambda2)

def compute_bounds(triangle):
    min_x = int(np.min(triangle[:,0]))
    max_x = int(np.max(triangle[:,0]))
    min_y = int(np.min(triangle[:,1]))
    max_y = int(np.max(triangle[:,1]))
    
    return min_x, max_x, min_y, max_y

def texture_map(texture, uv):
    u, v = uv
    x = int(u * (texture.get_width() - 1))
    y = int(v * (texture.get_height() - 1))
    return texture.get_at((x,y))

zbuffer = np.full((screenWidth, screenHeight), np.inf)

def limit_value(v): 
    return max(0.0, min(1.0, v))

def get_uv_at_point(v1,v2,v3,x,y):
    d=((v2.position[1]-v3.position[1])*(v1.position[0]-v3.position[0])
      +(v3.position[0]-v2.position[0])*(v1.position[1]-v3.position[1]))
    w1=((v2.position[1]-v3.position[1])*(x-v3.position[0])
      +(v3.position[0]-v2.position[0])*(y-v3.position[1]))/d
    w2=((v3.position[1]-v1.position[1])*(x-v3.position[0])
      +(v1.position[0]-v3.position[0])*(y-v3.position[1]))/d
    w3=1-w1-w2
    return w1,w2,w3

def scanline_render_vertex_texture_fill_triangle(screen,vertices,texture):
    # y좌표를 기준으로 정렬
    if vertices[0].position[1] > vertices[1].position[1]:
        vertices[0], vertices[1] = vertices[1], vertices[0]
    if vertices[0].position[1] > vertices[2].position[1]:
        vertices[0], vertices[2] = vertices[2], vertices[0]
    if vertices[1].position[1] > vertices[2].position[1]:
        vertices[1], vertices[2] = vertices[2], vertices[1]
    
    v1, v2, v3 = vertices
    v1_X = int(v1.position[0]); v1_Y = int(v1.position[1]); v1_Z = (v1.position[2])
    v2_X = int(v2.position[0]); v2_Y = int(v2.position[1]); v2_Z = (v2.position[2])
    v3_X = int(v3.position[0]); v3_Y = int(v3.position[1]); v3_Z = (v3.position[2])
    
    # 각 변의 기울기를 계산합니다.
    slope_12 = (v2_X - v1_X) / (v2_Y - v1_Y) if v2_Y != v1_Y else 0
    slope_13 = (v3_X - v1_X) / (v3_Y - v1_Y) if v3_Y != v1_Y else 0
    slope_23 = (v3_X - v2_X) / (v3_Y - v2_Y) if v3_Y != v2_Y else 0
    
    # z값에 대한 기울기를 계산
    z_slope_12 = (v2_Z - v1_Z) / (v2_Y - v1_Y) if v2_Y != v1_Y else 0
    z_slope_13 = (v3_Z - v1_Z) / (v3_Y - v1_Y) if v3_Y != v1_Y else 0
    z_slope_23 = (v3_Z - v2_Z) / (v3_Y - v2_Y) if v3_Y != v2_Y else 0
    
    for y in range(v1_Y, v3_Y + 1):
        if y <= v2_Y :
            x1 = v1_X + (y - v1_Y ) * slope_13
            z1 = v1_Z + (y - v1_Y) * z_slope_13
            x2 = v1_X + (y - v1_Y ) * slope_12
            z2 = v1_Z + (y - v1_Y) * z_slope_12
        else:
            x1 = v1_X + (y - v1_Y ) * slope_13
            z1 = v1_Z + (y - v1_Y) * z_slope_13
            x2 = v2_X + (y - v2_Y) * slope_23
            z2 = v2_Z + (y - v2_Y) * z_slope_23
        if x1 < x2:
            x_left = int(x1); x_right = int(x2)
            z_left = (z1); z_right = (z2)
        else:
            x_left = int(x2); x_right = int(x1)
            z_left = (z2); z_right = (z1)
            
        # z의 기울기 계산
        z_slope = (z_right - z_left) / (x_right - x_left) if x_right != x_left else 0

        for x in range(x_left, x_right): # left에서 right까지 렌더링
            z = z_left + (x - x_left) * z_slope
            
            if z < zbuffer[x, y]:
                # Z-버퍼를 업데이트하고 픽셀을 그린다.
                zbuffer[x, y] = z
                u, v, w = get_uv_at_point(v1,v2,v3,x,y)
                uv = u * v1.uv + v * v2.uv + w * v3.uv
                uv[0] = limit_value(uv[0])
                uv[1] = limit_value(uv[1])
                tex_x = min(int(uv[0] * texture.get_width()), texture.get_width() - 1)
                tex_y = min(int(uv[1] * texture.get_height()), texture.get_height() - 1)
                tex_color = texture.get_at((tex_x, tex_y))
                screen.set_at((x, y), tex_color)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(BACKGROUND_COLR)
    
    rect1_model_matrix = ModelMatrix([0, 0, 0], [1, 1, 1], [0, 0, 0])
    # Update
    proj_view_model_matrix = np.matmul(projection_matrix, np.matmul(view_matrix, rect1_model_matrix))
    screen_data = []
    for vertex in vertices:
        # 3D 좌표계에서 2D 좌표계로 변환
        clip = np.matmul(proj_view_model_matrix, np.append(vertex.position, 1))
        # NDC 좌표 변환
        ndc = clip / clip[3]
        # 화면 좌표 변환
        scr = np.matmul(viewport_matrix, ndc)
        screen_data.append((scr[0], scr[1], scr[2], vertex.uv))
        
    zbuffer = np.full((screenWidth, screenHeight), np.inf)
    
    for face in trifaces:
        tri = []
        for index in face:
            sx, sy, sz, uv = screen_data[index]
            tri.append(Vertex([sx, sy, sz], uv))
        scanline_render_vertex_texture_fill_triangle(screen, tri, texture)
        
    pygame.display.flip()
pygame.quit()