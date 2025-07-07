import pygame
import numpy as np
import math
pygame.init()

screenWidth = 1280
screenHeight = 720

BACKGROUND_COLR = (120, 120, 120)
LINE_COLOR = (0,0,0)
RedColor = (255, 0, 0)
BlueColor = (0, 0, 255)
GreenColor = (0, 255, 0)
PurpleColor = (255, 0, 255)

screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Graphics Programming")
running = True

def DrawLine(x1, y1, x2, y2, Color) :
    x, y = x1, y1
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    if dx > dy:
        p = -2 * dy + dx
        while True:
            screen.set_at((x, y), Color)       
            if x == x2 and y == y2:
                break
            x += sx
            if p < 0:
                y += sy
                p += -2*dy + 2*dx
            else:
                p += -2*dy
    else:
        p = -2 * dx + dy
        while True:
            screen.set_at((x, y), Color)
            if x == x2 and y == y2:
                break
            y += sy
            if p < 0:
                x += sx
                p += -2*dx + 2*dy
            else:
                p += -2 * dx


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLR)
    
    # 중간 선 그리기
    DrawLine(0, 0, screenWidth, screenHeight, (0, 0, 0))
    
    # 기울기가 1보다 작을 때 x축 값을 1씩 증가
    DrawLine(0, 0, 400, 100, GreenColor)
    
    # 기울기가 1보다 클 때 Y축 값을 1씩 증가
    DrawLine(0, 0, 100, 400, BlueColor)
    
    # 수직 직선 그리기
    DrawLine(0, 0, 0, 400, RedColor)
    
    # (x1, y1)이 (x2, y2) 보다 클 때
    DrawLine(300, 400, 0, 0, PurpleColor)

    pygame.display.flip()

pygame.quit()