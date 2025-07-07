import pygame
import numpy as np
import math
pygame.init()

screenWidth = 1280
screenHeight = 720

BACKGROUND_COLR = (0, 0, 0)
LINE_COLOR = (255,255,255)
RedColor = (255, 0, 0)
BlueColor = (0, 0, 255)
GreenColor = (0, 255, 0)
PurpleColor = (255, 0, 255)

screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Graphics Programming")
running = True


# ===== Model Matrix  =====
Tx, Ty, Tz = 0, 0, 0
AngleX = math.radians(30)
AngleY = math.radians(30)
AngleZ = math.radians(30)
Sx, Sy, Sz = 200, 200, 200

# 동차 행렬 정의
# 위치
T = np.array([
    [1, 0, 0, Tx],
    [0, 1, 0, Ty],
    [0, 0, 1, Tz],
    [0, 0, 0, 1]
])

# 회전 x, y, z
CosX = math.cos(AngleX)
SinX = math.sin(AngleX)
Rx = np.array([
    [1,  0,     0,   0],
    [0, CosX, -SinX, 0],
    [0, SinX,  CosX, 0],
    [0,  0,     0,   1]
])

CosY = math.cos(AngleY)
SinY = math.sin(AngleY)
Ry = np.array([
    [CosY,  0, SinY, 0],
    [ 0,    1,  0,   0],
    [-SinY, 0, CosY, 0],
    [  0 ,  0,  0,   1]
])

CosZ = math.cos(AngleZ)
SinZ = math.sin(AngleZ)
Rz = np.array([
    [CosZ, -SinZ, 0, 0],
    [SinZ, CosZ,  0, 0],
    [ 0,    0,    1, 0],
    [ 0,    0,    0, 1]
])

R = Rz @ Ry @ Rx

# 크기
S = np.array([
    [Sx, 0, 0, 0],
    [0, Sy, 0, 0],
    [0, 0, Sz, 0],
    [0, 0, 0,  1]
])

Model = T @ R @ S
# =========================


# ====== view matrix ====== 
cam_pos = np.array([0, 0, 500])   # 카메라 위치 
target_pos = np.array([0, 0, 0])  # 바라보는 위치
up = np.array([0, 1, 0])          # 카메라의 기준이 되는 위쪽 방향
 
view_z = (target_pos - cam_pos)
view_z = view_z / np.linalg.norm(view_z)

view_x = np.cross(up, view_z)
view_x = view_x / np.linalg.norm(view_x)

view_y = np.cross(view_z, view_x)
view_y = view_y/ np.linalg.norm(view_y)
 
cam_inv_model_matrix  = np.array([
    [view_x[0], view_x[1], view_x[2], -np.dot(view_x, cam_pos)],
    [view_y[0], view_y[1], view_y[2], -np.dot(view_y, cam_pos)],
    [view_z[0], view_z[1], view_z[2], -np.dot(view_z, cam_pos)],
    [  0,         0,         0,                    1          ]
])
 
rotation_y_180 = np.array([
    [-1, 0,  0, 0],
    [ 0, 1,  0, 0],
    [ 0, 0, -1, 0],
    [ 0, 0,  0, 1]
])
# y축 180를 회전시켜서 x,y축이 수학적인 2차원 좌표계처럼 보이게 수정
view_matrix = np.matmul(rotation_y_180, cam_inv_model_matrix)
# =========================


# === Projection Matrix ===
near = 0.1     # near Plane 거리
far = 100      # far Plane 거리
d = 1.0 / math.tan(90/2)
k = screenWidth / screenHeight

Projection = np.array([
    [d/k,  0,           0,                                         0],
    [ 0,   d,           0,                                         0],
    [ 0,   0, (near+far)/(near-far), (2 * near * far) / (near - far)],
    [ 0,   0,           -1,                                        0]
])
# =========================


# MVP
mvp = Projection @ view_matrix @ Model


# 큐브의 정점
CubeVertex = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
]

# 큐브의 선 연결 정보
CubeEdges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# 큐브 대각선
DiagonalEdged = [
    (0, 2),
    (4, 6),
    (0, 7),
    (1, 6),
    (0, 5),
    (3, 6),
]


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLR)
    
    result = []
    for x, y, z in CubeVertex:
        v_Local = np.array([x, y, z, 1])
        Clip  = mvp @ v_Local
        if Clip[3] == 0 : 
            continue
        
        # ndc
        ndc = Clip / Clip[3]
        
        # 뷰포트 변환
        sx = int((ndc[0] + 1.0) * 0.5 * screenWidth)
        sy = int((1.0 - ndc[1]) * 0.5 * screenHeight)
        result.append((sx, sy))
        
    # 정육면체 edges 그리기
    for i0, i1 in CubeEdges:
        pygame.draw.line(screen, LINE_COLOR, result[i0], result[i1])
        
    # 정육면체 대각선 그리기
    for i0, i1 in DiagonalEdged:
        pygame.draw.line(screen, GreenColor, result[i0], result[i1])
        
    
    
    pygame.display.update()

pygame.quit()