import pygame
import numpy as np
import math
pygame.init()

screenWidth = 1280
screenHeight = 720

BACKGROUND_COLR = (255, 255, 255)
LINE_COLOR = (0,0,0)
RedColor = (255, 0, 0)
BlueColor = (0, 0, 255)
GreenColor = (0, 255, 0)
PurpleColor = (255, 0, 255)

screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Graphics Programming")
running = True

CameraX, CameraY = 100, 100

def GetWorldPosX(gotX):
    return screenWidth / 2 + (gotX - CameraX)

def GetWorldPosY(gotY):
    return screenHeight / 2 - (gotY - CameraY)

def DrawCoordinatePlane() :
    Grid = 10
    y = 0
    while y < screenHeight:
        y += Grid
        pygame.draw.line(screen, LINE_COLOR , (0, y), (screenWidth, y), 1)
        
    x = 0
    while x < screenWidth:
        x += Grid
        pygame.draw.line(screen, LINE_COLOR , (x, 0), (x, screenHeight), 1)
        
    pygame.draw.line(screen, LINE_COLOR , (0, GetWorldPosY(0)), (screenWidth, GetWorldPosY(0)), 2)
    pygame.draw.line(screen, LINE_COLOR , (GetWorldPosX(0), 0), (GetWorldPosX(0), screenHeight), 2)

def DrawRect(x, y, width, height, Color) :
    left_top_x = GetWorldPosX(x) - width / 2
    left_top_y = GetWorldPosY(y) - height / 2
    DrawRectPoint = pygame.Rect(left_top_x, left_top_y, width, height)
    pygame.draw.rect(screen, Color, DrawRectPoint)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLR)
    
    DrawCoordinatePlane()
    
    DrawRect(0, 0, 100, 100, RedColor)
    DrawRect(300, 350, 100, 100, RedColor)
    DrawRect(-300, -300, 100, 100, RedColor)
    
    pygame.display.update()

pygame.quit()