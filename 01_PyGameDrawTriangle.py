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

def DrawTriangle(Vertices, color) : 
    if Vertices[0][1] > Vertices[1][1]:
        Vertices[0], Vertices[1] = Vertices[1], Vertices[0]

    if Vertices[0][1] > Vertices[2][1]:
        Vertices[0], Vertices[2] = Vertices[2], Vertices[0]

    if Vertices[1][1] > Vertices[2][1]:
        Vertices[1], Vertices[2] = Vertices[2], Vertices[1]

    v1, v2, v3 = Vertices

    def calculate_slope(v_start, v_end):
        if v_end[1] != v_start[1]:
            return (v_end[0] - v_start[0]) / (v_end[1] - v_start[1])
        else:
            return None

    slope_12 = calculate_slope(v1, v2)
    slope_13 = calculate_slope(v1, v3)
    slope_23 = calculate_slope(v2, v3)  

    for y in range(int(v1[1]), int(v3[1]) + 1):
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
       
        x_left = math.floor(min(x_start, x_end))
        x_right = math.ceil(max(x_start, x_end))
        
        for x in range(x_left, x_right+1):
             screen.set_at((x, y), color)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLR)
    
    Vertices = [
        [100, 50], 
        [50, 200],
        [150, 250]
    ]
    DrawTriangle(Vertices, RedColor)
    
    Vertices = [
        [250, 50], 
        [300, 250],
        [450, 150]
    ]
    DrawTriangle(Vertices, BlueColor)
    
    Vertices = [
        [130, 300], 
        [50, 300],
        [150, 450]
    ]
    DrawTriangle(Vertices, GreenColor)
    
    

    pygame.display.flip()

pygame.quit()