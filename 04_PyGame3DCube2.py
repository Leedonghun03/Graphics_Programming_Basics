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

cube_size = 50
half_size = cube_size / 2
tripoint2 = np.array([
    [-1* half_size, -1* half_size, 1* half_size, 1],
    [ 1* half_size, -1* half_size, 1* half_size, 1],
    [ 1* half_size, 1* half_size, 1* half_size, 1],
    [-1* half_size, 1* half_size, 1* half_size, 1],
    [-1* half_size, -1* half_size, -1* half_size, 1],
    [ 1* half_size, -1* half_size, -1* half_size, 1],
    [ 1* half_size, 1* half_size, -1* half_size, 1],
    [-1* half_size, 1* half_size, -1* half_size, 1]
])

# 삼각형을 이루는 정점 인덱스 구축
trifaces = [
    [0, 1, 2], # Front
    [2, 3, 0], # Front
    [7, 6, 5], # Back
    [5, 4, 7], # Back
    [4, 5, 1], # Bottom
    [1, 0, 4], # Bottom
    [3, 2, 6], # Top
    [6, 7, 3], # Top
    [4, 0, 3], # Left
    [3, 7, 4], # Left
    [1, 5, 6], # Right
    [6, 2, 1] # Right
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
cam_pos = np.array([150, 200, 300])

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

def scanline_fill_triangle(screen,color,vertices):    

    # y좌표를 기준으로 정렬    
    if vertices[0][1] > vertices[1][1]:
        vertices[0], vertices[1] = vertices[1], vertices[0]

    if vertices[0][1] > vertices[2][1]:
        vertices[0], vertices[2] = vertices[2], vertices[0]

    if vertices[1][1] > vertices[2][1]:
        vertices[1], vertices[2] = vertices[2], vertices[1]

    v1, v2, v3 = vertices

    # 각 변의 기울기를 계산합니다.
    def calculate_slope(v_start, v_end):
        if v_end[1] != v_start[1]:
            return (v_end[0] - v_start[0]) / (v_end[1] - v_start[1])
        else:
            return None  # 수평선일 경우 기울기가 없음을 나타냅니다.

    slope_12 = calculate_slope(v1, v2)
    slope_13 = calculate_slope(v1, v3)
    slope_23 = calculate_slope(v2, v3)  

    # y 좌표에 따라 x 좌표를 계산하고 화면에 픽셀을 채웁니다.
    for y in range(int(v1[1]), int(v3[1]) + 1):  # range : start, stop-1
        if y <= v2[1]:
            if slope_12 is not None:
                x_start = slope_12 * (y - v1[1]) + v1[0]
            else:
                x_start = v2[0]
        else:
            if slope_23 is not None:
                x_start = slope_23 * (y - v2[1]) + v2[0]
            else:
                x_start = v2[0]

        if slope_13 is not None:
            x_end = slope_13 * (y - v1[1]) + v1[0]
        else:
            x_end = v3[0]        
       
        x_left, x_right =  math.floor(min(x_start, x_end)), math.ceil(max(x_start, x_end))            
             
        for x in range(x_left, x_right+1):
             screen.set_at((x, y), color)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLR)
    
    rect1_model_matrix = ModelMatrix([10, 10, 10], [1, 1, 1], [0, 0, 0])
    # Update
    proj_view_model_matrix = np.matmul(projection_matrix,np.matmul(view_matrix, rect1_model_matrix))
    transform_points = []
    for vertex in tripoint:
        # 3D 좌표계에서 2D 좌표계로 변환
        v = np.matmul(proj_view_model_matrix, vertex)
        # NDC 좌표 변환
        v /= v[3]
        # 화면 좌표 변환
        v = np.matmul(viewport_matrix, v)
        transform_points.append(v[:3])
    
    for face in trifaces:
        face_points = []
        for j in face:
            face_points.append(transform_points[j][:2])
        scanline_fill_triangle(screen, BlueColor, face_points)
    
    rect2_model_matrix = ModelMatrix([-150, 150, -10], [1, 1, 1], [0, 0, 0])
    proj_view_model_matrix2 = np.matmul(projection_matrix,np.matmul(view_matrix, rect2_model_matrix))
    transform_points = []
    for vertex in tripoint2:
        # 3D 좌표계에서 2D 좌표계로 변환
        v = np.matmul(proj_view_model_matrix2, vertex)
        # NDC 좌표 변환
        v /= v[3]
        # 화면 좌표 변환
        v = np.matmul(viewport_matrix, v)
        transform_points.append(v[:3])
    
    for face in trifaces:
        face_points = []
        for j in face:
            face_points.append(transform_points[j][:2])
        scanline_fill_triangle(screen, GreenColor, face_points)
    
    pygame.display.flip()
pygame.quit()