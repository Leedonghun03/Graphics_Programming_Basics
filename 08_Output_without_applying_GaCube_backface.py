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
cam_pos = np.array([0, 0, 300])

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
    y = int((1-v) * (texture.get_height() - 1))
    return texture.get_at((x,y))

def render_triangle(screen, textrue, triangle, triangle_uv, min_x, max_x, min_y, max_y):
    for x in range(int(min_x), int(max_x + 1)):
        for y in range(int(min_y), int(max_y + 1)):
            point = np.array([x,y])
            
            lambda1, lambda2, lambda3 = baryentric_coords_ext(triangle, point)
            
            if lambda1 >= -1.15e-16 and lambda2 >= -1.15e-16 and lambda3 >= -1.15e-16 :
                uv = lambda1 * triangle_uv[0] + lambda2 * triangle_uv[1] + lambda3 * triangle_uv[2]
                color = texture_map(textrue, uv)
                screen.set_at((x, y), color)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(BACKGROUND_COLR)
    
    rect1_model_matrix = ModelMatrix([0, 0, 0], [1, 1, 1], [0, 0, 0])
    # Update
    proj_view_model_matrix = np.matmul(projection_matrix,np.matmul(view_matrix, rect1_model_matrix))
    transform_points = []
    for vertex in vertices:
        pos = np.append(vertex.position, 1)
        # 3D 좌표계에서 2D 좌표계로 변환
        v = np.matmul(proj_view_model_matrix, pos)
        # NDC 좌표 변환
        v /= v[3]
        # 화면 좌표 변환
        v = np.matmul(viewport_matrix, v)
        transform_points.append(v[:2])
    
    for face in trifaces:
        pts_list = []
        for i in face:
            pts_list.append(transform_points[i][:2])
        tri_pts = np.array(pts_list)
        
        tri_list = []
        for i in face:
            tri_list.append(vertices[i].uv)
        tri_uvs = np.array(tri_list)
        
        min_x, max_x, min_y, max_y = compute_bounds(tri_pts)
        
        render_triangle(screen, texture,
                        tri_pts,    # 2D 좌표
                        tri_uvs,    # UV 좌표
                        min_x, max_x, 
                        min_y, max_y)
        
    pygame.display.flip()
pygame.quit()