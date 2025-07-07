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

triangle = np.array([(200,200), (200, 400), (400, 200)])
triangle_uv = np.array([(0, 1), (0,0), (1,1)])

triangle2 = np.array([(400, 200), (200, 400), (400, 400)])
triangle2_uv = np.array([(1, 1), (0, 0), (1, 0)])

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
    min_x = min(triangle[:, 0])
    max_x = max(triangle[:, 0])
    min_y = min(triangle[:, 1])
    max_y = max(triangle[:, 1])
    
    return min_x, max_x, min_y, max_y

def texture_map(texture, uv):
    u, v = uv
    x = int(u * (texture.get_width() - 1))
    y = int((1-v) * (texture.get_height() - 1))
    return texture.get_at((x,y))

def render_triangle(screen, textrue, triangle, triangle_uv, min_x, max_x, min_y, max_y):
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            point = np.array([x,y])
            
            lambda1, lambda2, lambda3 = baryentric_coords_ext(triangle, point)
            
            if lambda1 >= -1.15e-16 and lambda2 >= -1.15e-16 and lambda3 >= -1.15e-16 :
                uv = lambda1 * triangle_uv[0] + lambda2 * triangle_uv[1] + lambda3 * triangle_uv[2]
                color = texture_map(textrue, uv)
                screen.set_at((x, y), color)
                
                
bounds = compute_bounds(triangle)    
bounds2 = compute_bounds(triangle2)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLR)
    
    render_triangle(screen, texture, triangle, triangle_uv, *bounds)
    render_triangle(screen, texture, triangle2, triangle2_uv, *bounds2)
    
    pygame.display.flip()
pygame.quit()