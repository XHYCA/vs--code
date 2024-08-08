import pygame  
import time  
import random  

# 初始化pygame  
pygame.init()  

# 定义颜色  
white = (255, 255, 255)  
yellow = (255, 255, 102)  
black = (0, 0, 0)  
red = (213, 50, 80)  
green = (0, 255, 0)  
blue = (50, 153, 213)  

# 设置游戏窗口的宽度和高度  
width = 600  
height = 400  

# 创建游戏窗口  
screen = pygame.display.set_mode((width, height))  
pygame.display.set_caption('贪吃蛇游戏')  

# 游戏时钟  
clock = pygame.time.Clock()  

snake_block = 10  
snake_speed = 15  

# 字体样式  
font_style = pygame.font.SysFont("bahnschrift", 25)  
score_font = pygame.font.SysFont("comicsansms", 35)  


def our_snake(snake_block, snake_list):  
    for x in snake_list:  
        pygame.draw.rect(screen, black, [x[0], x[1], snake_block, snake_block])  


def message(msg, color):  
    mesg = font_style.render(msg, True, color)  
    screen.blit(mesg, [width / 6, height / 3])  


def gameLoop():  # 创建一个函数来运行游戏  
    game_over = False  
    game_close = False  

    x1 = width / 2  # 蛇的起始位置  
    y1 = height / 2  

    x1_change = 0  
    y1_change = 0  

    snake_List = []  
    Length_of_snake = 1  

    foodx = round(random.randrange(0, width - snake_block) / 10.0) * 10.0  
    foody = round(random.randrange(0, height - snake_block) / 10.0) * 10.0  

    while not game_over:  # 游戏主循环  
        while game_close == True:  # 显示游戏结束信息  
            screen.fill(blue)  
            message("你输了! Q-退出 C-重玩", red)  
            pygame.display.update()  

            for event in pygame.event.get():  
                if event.type == pygame.KEYDOWN:  
                    if event.key == pygame.K_q:  
                        game_over = True  
                        game_close = False  
                    if event.key == pygame.K_c:  
                        gameLoop()  

        for event in pygame.event.get():  
            if event.type == pygame.QUIT:  
                game_over = True  
            if event.type == pygame.KEYDOWN:  
                if event.key == pygame.K_LEFT:  
                    x1_change = -snake_block  
                    y1_change = 0  
                elif event.key == pygame.K_RIGHT:  
                    x1_change = snake_block  
                    y1_change = 0  
                elif event.key == pygame.K_UP:  
                    y1_change = -snake_block  
                    x1_change = 0  
                elif event.key == pygame.K_DOWN:  
                    y1_change = snake_block  
                    x1_change = 0  

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:  
            game_close = True  

        x1 += x1_change  
        y1 += y1_change  
        screen.fill(blue)  

        pygame.draw.rect(screen, green, [foodx, foody, snake_block, snake_block])  
        snake_Head = []  
        snake_Head.append(x1)  
        snake_Head.append(y1)  
        snake_List.append(snake_Head)  
        if len(snake_List) > Length_of_snake:  
            del snake_List[0]  

        for x in snake_List[:-1]:  
            if x == snake_Head:  
                game_close = True  

        our_snake(snake_block, snake_List)  

        pygame.display.update()  

        if x1 == foodx and y1 == foody:  
            foodx = round(random.randrange(0, width - snake_block) / 10.0) * 10.0  
            foody = round(random.randrange(0, height - snake_block) / 10.0) * 10.0  
            Length_of_snake += 1  

        clock.tick(snake_speed)  

    pygame.quit()  
    quit()  


gameLoop()