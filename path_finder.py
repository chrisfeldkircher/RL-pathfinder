import pygame
from pygame.locals import *
import numpy as np
import copy
import csv
import os
import re
from queue import PriorityQueue
from tkinter import Tk
from tkinter import filedialog as fd
import platform
import sys

class Simulation:
    def __init__(self, filename = '', load_map = False, load_qmap = False):
        self.screen = None
        self.dim = self.width, self.height = 800, 600
        self.fps = 30
        self.size = 16#20#16 #Size of the Grid-matrix
        self.grid_size = 450 #resolution grid
        self.origin = (300,100)
        self.grid = [[0 for i in range(self.size)] for j in range(self.size)] #empty grid 
        self.backup_grid = [[0 for i in range(self.size)] for j in range(self.size)]
        self.q_map = [[{'up':0,'down':0,'right':0,'left':0} for i in range(self.size)] for j in range(self.size)]
        self.moves= ['right', 'down', 'up', 'left']
        self.iterations = 1500#1320 #Higher this to raise the exploration time of the ai (random mode is gonna last longer)
        self.episode = 0
        self.discount = 0.97 #
        self.learn_rate = 0.8 #
        self.target_reward = 100 #Reward for finding the goal
        self.gift_reward = 25 #Reward for the purple gifts
        self.epsilon = 50 #150 #Lower this to raise the exploration time of the ai (random mode is gonna last longer)
        self.action = ''
        self.reward = {}
        self.actions = 0
        self.action_limit = 700 #300 #Limit of moves per episode
        self.episode_limit = 90 #Limit of the episodes
        self.mode = ''
        self.num_walls = int((self.size*self.size)/3) #Number of obstacles
        self.num_gift = 3 #Number of gifts
        self.dist_reward = 0 #Do not change
        self.dist_rew = 0 #Reward for getting closer (no need to use)
        self.distance_old = 0
        self.neg_reward = -100 #Negative reward if q_map is initialized by self.init_q_map()
        self.stop = False
        self.p_queue = PriorityQueue() #PriorityQueue
        self.f_val = {(i,j):float('inf') for i in range(self.size) for j in range(self.size)} #h_val + g_val -> init. cells with high weight
        self.g_val = {(i,j):float('inf') for i in range(self.size) for j in range(self.size)} #actual distance start, cell
        self.D = 1 #Minimal cost for one step
        self.pl_goal_distance = 12 #Minimal distance between start and goal cell (Is used by self.random_grid)
        self.goal = (0,0) #position goal, gets assigned by self.random_grid
        self.start = (0,0) #position start, gets assigned by self.random_grid
        self.current = (0,0) #position current, gets assigned by self.random_grid
        self.options = [[{'up':0,'down':0,'right':0,'left':0} for i in range(self.size)] for j in range(self.size)]
        self.path = [] #a* - path
        self.show_path = False 
        self.show_ai_path = False
        self.ai_path = []
        self.copy_path = False
        self.check_epsiode = 35 #Check episode
        self.load_qmap = load_qmap #Change to True if a qmap should be loaded
        self.saved_qmap_name = filename #Name of the saved q_map which should be loaded
        self.ld_map = load_map #Change to True if a map should be loaded
        self.saved_map_name = filename #Name of the map which should be loaded
        self.save_qmap = False
        self.map_name = ''
        #self.general_txt = ''
        self.root = Tk()
        self.steps_ai = self.action_limit

    def init_q_map(self):
        valide_cells = [0,2,3,4]
        for i in range(len(self.grid)):
            for j in  range(len(self.grid[i])):
                if self.grid[i][j] != 1:
                    if i != 0: #not at upper boundary
                        if i != len(self.grid)-1: #not at lower boundary
                            if j != 0: #not at left boundary
                                if j != len(self.grid[i])-1: #not at right,left,upper,lower boundary
                                    for key in self.moves:
                                        if self.q_map[i][j][key] != -self.target_reward:
                                            self.q_map[i][j][key] = 0

                                else: #not at left,upper,lower boundary
                                    if self.q_map[i][j][key] != self.neg_reward:
                                        self.q_map[i][j][key] = 0

                                    self.q_map[i][j]['right'] = self.neg_reward  
                            else: #at left boundary
                                if j != len(self.grid[i])-1: #not at upper,lower boundary
                                    if self.q_map[i][j][key] != self.neg_reward:
                                        self.q_map[i][j][key] = 0
                                        
                                    self.q_map[i][j]['left'] = self.neg_reward   
                                else: #not at upper,lower boundary
                                    self.q_map[i][j]['up'] = 0
                                    self.q_map[i][j]['down'] = 0
                                    self.q_map[i][j]['left'] = self.neg_reward
                                    self.q_map[i][j]['right'] = self.neg_reward 
                        else: #at lower boundary
                            if j != 0: #not at left boundary
                                if j != len(self.grid[i])-1: #not at upper,right,left boundary
                                    for key in self.moves:
                                        if self.q_map[i][j][key] != self.neg_reward:
                                            self.q_map[i][j][key] = 0

                                        self.q_map[i][j]['down'] = self.neg_reward  
                                else: #not at upper,left boundary
                                    for key in self.moves:
                                        if self.q_map[i][j][key] != self.neg_reward:
                                            self.q_map[i][j][key] = 0

                                    self.q_map[i][j]['right'] = self.neg_reward
                                    self.q_map[i][j]['down'] = self.neg_reward  
                            else: #at left boundary
                                if j != len(self.grid[i])-1: #not at upper, right boundary
                                    self.q_map[i][j]['up'] = 0
                                    self.q_map[i][j]['right'] = 0 
                                    self.q_map[i][j]['down'] = self.neg_reward
                                    self.q_map[i][j]['left'] = self.neg_reward  
                                else: #not at upper boundary
                                    self.q_map[i][j]['up'] = 0
                                    self.q_map[i][j]['down'] = self.neg_reward
                                    self.q_map[i][j]['left'] = self.neg_reward
                                    self.q_map[i][j]['right'] = self.neg_reward
                    else: #at the upper boundary
                        if i != len(self.grid)-1: #not at the lower boundary
                            if j != 0: #not at the lower,left boundary
                                if j != len(self.grid[i])-1: #not at the lower,right,left boundary
                                    for key in self.moves:
                                        if self.q_map[i][j][key] != self.neg_reward:
                                            self.q_map[i][j][key] = 0 

                                        self.q_map[i][j]['up'] = self.neg_reward 
                                else: #not at the lower,left boundary
                                    self.q_map[i][j]['down'] = 0 
                                    self.q_map[i][j]['left'] = 0
                                    self.q_map[i][j]['right'] = self.neg_reward
                                    self.q_map[i][j]['up'] = self.neg_reward  
                            else: #at the left boundary
                                if j != len(self.grid[i])-1: #not at the lower,right boundary
                                    self.q_map[i][j]['down'] = 0 
                                    self.q_map[i][j]['right'] = 0
                                    self.q_map[i][j]['left'] = self.neg_reward
                                    self.q_map[i][j]['up'] = self.neg_reward   
                                else: #at the right boundary
                                    self.q_map[i][j]['down'] = 0 
                                    self.q_map[i][j]['up'] = self.neg_reward 
                                    self.q_map[i][j]['right'] = self.neg_reward 
                                    self.q_map[i][j]['left'] = self.neg_reward 
                        else: #at the lower boundary
                            if j != 0: #not at the left boundary
                                if j != len(self.grid[i])-1: #not at the left, right boundary
                                    self.q_map[i][j]['right'] = 0 
                                    self.q_map[i][j]['left'] = 0
                                    self.q_map[i][j]['up'] = self.neg_reward 
                                    self.q_map[i][j]['down'] = self.neg_reward  
                                else: #at the right boundary
                                    self.q_map[i][j]['left'] = 0
                                    self.q_map[i][j]['right'] = self.neg_reward 
                                    self.q_map[i][j]['up'] = self.neg_reward 
                                    self.q_map[i][j]['down'] = self.neg_reward   
                            else: #at the left boundary
                                if j != len(self.grid[i])-1: #not at the right bundary
                                    self.q_map[i][j]['right'] = 0
                                    self.q_map[i][j]['left'] = self.neg_reward 
                                    self.q_map[i][j]['up'] = self.neg_reward 
                                    self.q_map[i][j]['down'] = self.neg_reward    
                else:
                    if i != 0: #mark upper cell
                        if self.grid[i-1][j] in valide_cells:
                            self.q_map[i-1][j]['down'] = self.neg_reward
                    if i != len(self.grid)-1: #mark lower cell
                        if self.grid[i+1][j] in valide_cells:
                            self.q_map[i+1][j]['up'] = self.neg_reward
                    if j != 0: #mark left cell
                        if self.grid[i][j-1] in valide_cells:
                            self.q_map[i][j-1]['right'] = self.neg_reward
                    if j != len(self.grid[i])-1: #mark right cell
                        if self.grid[i][j+1] in valide_cells:
                            self.q_map[i][j+1]['left'] = self.neg_reward

    def random_grid(self):
        for _ in range(self.num_walls):
            self.grid[np.random.randint(0,self.size)][np.random.randint(0,self.size)] = 1

        for _ in range(self.num_gift):
            self.grid[np.random.randint(0,self.size)][np.random.randint(0,self.size)] = 4

        idx_player = (np.random.randint(0,self.size), np.random.randint(0,self.size))
        idx_goal = (np.random.randint(0,self.size), np.random.randint(0,self.size))
        
        while np.sqrt((idx_player[0] - idx_goal[0])**2 + (idx_player[1] - idx_goal[1])**2) < self.pl_goal_distance: 
            idx_player = (np.random.randint(0,self.size), np.random.randint(0,self.size))
            idx_goal = (np.random.randint(0,self.size), np.random.randint(0,self.size))

        self.grid[idx_player[0]][idx_player[1]] = 2
        self.grid[idx_goal[0]][idx_goal[1]] = 3
        
        self.start = (idx_player[0], idx_player[1])
        self.goal = (idx_goal[0], idx_goal[1])

    def save_map(self, map_type):
        if platform.system() == 'Windows':
            file_path = os.path.realpath(__file__).split('\\') # for windows 
        else:
            file_path = os.path.realpath(__file__).split('/') #for mac
        file_path = '/'.join([file_path[i] for i in range(len(file_path)-1)])
        file_list = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        maps = []
        
        if map_type == "grid":
            for file in file_list:
                if re.search(fr'(map)(.*)',file):
                    maps.append(re.search(fr'(map)(.*)',file).group())

            if maps:
                maps = [a.split('.') for a in maps]
                maps = int(max([a[0][-1] for a in maps if a[0][-1].isnumeric()])) + 1
            else:
                maps = 1
            
            self.map_name = f'map{maps}'.split('.')[0]
            
            with open(os.path.join(file_path, f'map{maps}.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in self.backup_grid:
                    writer.writerow(row)
        
        else:
            for file in file_list:
                if re.search(fr'(qmap)(.*)',file):
                    maps.append(re.search(fr'(qmap)(.*)',file).group())
            
            if maps:
                maps = [a.split('.') for a in maps]
                maps = int(max([a[0][-1] for a in maps if a[0][-1].isnumeric()])) + 1
            else:
                maps = 1
                
            with open(os.path.join(file_path, f'qmap{maps}-{self.map_name}.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in self.q_map:
                    writer.writerow(row)

    def load_map(self, name, map_type):
        file_path = os.path.realpath(__file__).split('/')
        file_path = '/'.join([file_path[i] for i in range(len(file_path)-1)])
        loaded_map = []
        loaded_qmap = []
        i = 0
        
        if map_type == 'grid':
            map_name = name + '.csv'
            with open(os.path.join(file_path, f'{map_name}'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    loaded_map.append(list(map(int, row)))
                    if 2 in list(map(int, row)):
                        self.start = (i, list(map(int, row)).index(2))
                    if 3 in list(map(int, row)):
                        self.goal = (i, list(map(int, row)).index(3))
                    
                    i += 1
            
            self.size = len(loaded_map)
            return loaded_map
        
        else:
            map_name = name.split('-')[1] + '.csv'
            qmap_name = name + '.csv'
            
            with open(os.path.join(file_path, f'{qmap_name}'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    loaded_qmap.append([eval(k) for k in row])

            with open(os.path.join(file_path, f'{map_name}'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    loaded_map.append(list(map(int, row)))
                    if 2 in list(map(int, row)):
                        self.start = (i, list(map(int, row)).index(2))
                    if 3 in list(map(int, row)):
                        self.goal = (i, list(map(int, row)).index(3))
                    
                    i += 1
            
            self.size = len(loaded_map)
            
            return loaded_map, loaded_qmap

    def init(self):
        pygame.init()
        pygame.display.set_caption("Shortest-Path Reinforcement learning")

        self.screen = pygame.display.set_mode(self.dim)
        self.isRunning = True
        self.screen.fill((255,255,255))

        self.clock = pygame.time.Clock()
        if self.ld_map:
            self.map_name = self.saved_map_name
            self.grid = copy.deepcopy(self.load_map(self.map_name, 'grid')) #load map
        else:
            self.random_grid()
        
        if self.load_qmap:
            self.grid, self.q_map = copy.deepcopy(self.load_map(self.saved_qmap_name, 'qmap')) #load qmap
            self.iterations = 0

        '''
        self.grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 2, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 3],
        [0, 0, 0, 0, 1, 0, 0, 0, 0]]
        #1 = Obstacle
        #2 = Player
        #3 = Goal
        #4 = Bomb
        #5 = Treasure
        
        self.start = (1,1)
        self.goal = (7,8)
        self.size = 9
        '''


        self.backup_grid = copy.deepcopy(self.grid)
    
        #self.init_q_map()
        self.grid_options()
        
        self.a_star()

    def update_Q_values(self, indicies_new, cell_type, move, indicies_old):
        #Q1 = self.target_reward
        #Q2 = (1-self.learn_rate)*self.q_map[indicies_old[0]][indicies_old[1]][move] + self.learn_rate*(self.gift_reward+(self.discount**self.actions)*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)])
        #Q3 = (1-self.learn_rate)*self.q_map[indicies_old[0]][indicies_old[1]][move] + self.learn_rate*(0+(self.discount**self.actions)*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)])

        #Q-Learning
        Q1 = (1-self.learn_rate)*self.q_map[indicies_old[0]][indicies_old[1]][move] + self.learn_rate*(self.target_reward+self.discount*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)])
        Q2 = (1-self.learn_rate)*self.q_map[indicies_old[0]][indicies_old[1]][move] + self.learn_rate*(self.gift_reward+self.discount*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)])
        Q3 = (1-self.learn_rate)*self.q_map[indicies_old[0]][indicies_old[1]][move] + self.learn_rate*(0+self.discount*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)])

        #print(self.q_map[indicies_old[0]][indicies_old[1]][move])
        #MDP
        #Q1 = self.target_reward + (self.discount)*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)]
        #Q2 = Q1 = self.gift_reward + (self.discount)*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)]
        #Q3 = (self.discount)*self.q_map[indicies_new[0]][indicies_new[1]][max(self.q_map[indicies_new[0]][indicies_new[1]], key=self.q_map[indicies_new[0]][indicies_new[1]].get)]

        if cell_type == 3:
            return Q1 + self.learn_rate*self.dist_reward
        elif cell_type == 4:
            return Q2 + self.learn_rate*self.dist_reward
        else:
            return Q3 + self.learn_rate*self.dist_reward

    def learn(self):
        epsilon = self.iterations/(self.episode*self.epsilon+1)
        p = np.random.uniform(0.3, 1)#np.random.uniform(0, 1)
        action = ''
        moves = {'up': (self.current[0] - 1, self.current[1]), 'down': (self.current[0] + 1, self.current[1]), 'right': (self.current[0], self.current[1] + 1), 'left': (self.current[0], self.current[1] - 1)}
        options = []
        
        if self.q_map[self.start[0]][self.start[1]][max(self.q_map[0][self.start[1]], key=self.q_map[self.start[0]][self.start[1]].get)] > 20:
            self.iterations = 0
        
        if p < epsilon:
            self.mode = 'random'
            #action = np.random.choice(self.moves)
            
            options = [k for k in self.moves if moves[k] in self.path]
            
            if np.random.uniform(0, 0.9) < 0.4 and options:#if np.random.uniform(0, 0.9) < 0.4 and options:
                self.action = np.random.choice(options)
            else:
                self.action = np.random.choice(self.moves)
            
        else:
            self.mode = 'policy'
            for i in range(len(self.grid)):
                if 2 in self.grid[i]:
                    idx = self.grid[i].index(2)
                    self.action = max(self.q_map[i][idx], key=self.q_map[i][idx].get)

    def placeObjects(self):
        cellBorder = 6
        celldimX = celldimY = (self.grid_size/self.size) - (cellBorder*2)
        for row in range(len(self.grid)):
            for column in range(len(self.grid)):
                obj = self.grid[column][row]
                if(obj == 1):
                    self.drawObstacles(self.origin[0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + 1,
                        self.origin[1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + 1, celldimX, celldimY, (0,0,0))
                elif(obj == 2):
                    self.drawObstacles(self.origin[0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + 1,
                        self.origin[1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + 1, celldimX, celldimY, (0,0,255))
                elif(obj == 3):
                    self.drawObstacles(self.origin[0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + 1,
                        self.origin[1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + 1, celldimX, celldimY, (0,255,0))
                elif(obj == 4):
                    self.drawObstacles(self.origin[0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + 1,
                        self.origin[1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + 1, celldimX, celldimY, (255,0,255))
                elif(obj == 5):
                    self.drawObstacles(self.origin[0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + 1,
                        self.origin[1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + 1, celldimX, celldimY, (255,255,0))
                elif(obj == 6):
                    self.drawObstacles(self.origin[0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + 1,
                        self.origin[1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + 1, celldimX, celldimY, (0,255,255))
                elif(obj == 7):
                    self.drawObstacles(self.origin[0] + (celldimY*row) + cellBorder + (2*row*cellBorder) + 1,
                        self.origin[1] + (celldimX*column) + cellBorder + (2*column*cellBorder) + 1, celldimX, celldimY, (50,100,50))

    def drawObstacles(self, x, y, size_x, size_y, color):
        pygame.draw.rect(self.screen, color, (x, y, size_x, size_y))

    def drawGrid(self, dim):
        #Border of the grid
        pygame.draw.rect(self.screen, (0,0,0), pygame.Rect(self.origin[0], self.origin[1], self.grid_size, self.grid_size,),  2)
        
        #Cell size
        cell_size = self.grid_size/self.size

        for i in range(dim):
            pygame.draw.line(self.screen, (0,0,0), (self.origin[0] + (cell_size * i), self.origin[1]), (self.origin[0] + (cell_size * i), self.grid_size + self.origin[1]), 2)
        #Horiz
            pygame.draw.line(self.screen, (0,0,0), (self.origin[0], self.origin[1] + (cell_size*i)), (self.origin[0] + self.grid_size, self.origin[1] + (cell_size*i)), 2)
        
    def heuristic(self, pnt1, pnt2):
        #Manhattan distance is used!
        dx = abs(pnt1[0] - pnt2[0])
        dy = abs(pnt1[1] - pnt2[1])
        
        return (self.D * (dx + dy))
    
    def grid_options(self):
        valide_cells = [0,2,3,4]
        for i in range(len(self.grid)):
            for j in  range(len(self.grid[i])):
                if self.grid[i][j] != 1:
                    if i != 0: #not at upper boundary
                        if i != len(self.grid)-1: #not at lower boundary
                            if j != 0: #not at left boundary
                                if j != len(self.grid[i])-1: #not at right,left,upper,lower boundary
                                    for key in self.moves:
                                        if self.options[i][j][key] != -1:
                                            self.options[i][j][key] = 0

                                else: #not at left,upper,lower boundary
                                    if self.options[i][j][key] != -1:
                                        self.options[i][j][key] = 0

                                    self.options[i][j]['right'] = -1  
                            else: #at left boundary
                                if j != len(self.grid[i])-1: #not at upper,lower boundary
                                    if self.options[i][j][key] != -1:
                                        self.options[i][j][key] = 0
                                        
                                    self.options[i][j]['left'] = -1  
                                else: #not at upper,lower boundary
                                    self.options[i][j]['up'] = 0
                                    self.options[i][j]['down'] = 0
                                    self.options[i][j]['left'] = -1
                                    self.options[i][j]['right'] = -1 
                        else: #at lower boundary
                            if j != 0: #not at left boundary
                                if j != len(self.grid[i])-1: #not at upper,right,left boundary
                                    for key in self.moves:
                                        if self.options[i][j][key] != -1:
                                            self.options[i][j][key] = 0

                                        self.options[i][j]['down'] = -1 
                                else: #not at upper,left boundary
                                    for key in self.moves:
                                        if self.options[i][j][key] != -1:
                                            self.options[i][j][key] = 0

                                    self.options[i][j]['right'] = -1
                                    self.options[i][j]['down'] = -1  
                            else: #at left boundary
                                if j != len(self.grid[i])-1: #not at upper, right boundary
                                    self.options[i][j]['up'] = 0
                                    self.options[i][j]['right'] = 0 
                                    self.options[i][j]['down'] = -1
                                    self.options[i][j]['left'] = -1  
                                else: #not at upper boundary
                                    self.options[i][j]['up'] = 0
                                    self.options[i][j]['down'] = -1
                                    self.options[i][j]['left'] = -1
                                    self.options[i][j]['right'] = -1
                    else: #at the upper boundary
                        if i != len(self.grid)-1: #not at the lower boundary
                            if j != 0: #not at the lower,left boundary
                                if j != len(self.grid[i])-1: #not at the lower,right,left boundary
                                    for key in self.moves:
                                        if self.options[i][j][key] != -1:
                                            self.options[i][j][key] = 0 

                                        self.options[i][j]['up'] = -1 
                                else: #not at the lower,left boundary
                                    self.options[i][j]['down'] = 0 
                                    self.options[i][j]['left'] = 0
                                    self.options[i][j]['right'] = -1
                                    self.options[i][j]['up'] = -1  
                            else: #at the left boundary
                                if j != len(self.grid[i])-1: #not at the lower,right boundary
                                    self.options[i][j]['down'] = 0 
                                    self.options[i][j]['right'] = 0
                                    self.options[i][j]['left'] = -1
                                    self.options[i][j]['up'] = -1   
                                else: #at the right boundary
                                    self.options[i][j]['down'] = 0 
                                    self.options[i][j]['up'] = -1 
                                    self.options[i][j]['right'] = -1 
                                    self.options[i][j]['left'] = -1 
                        else: #at the lower boundary
                            if j != 0: #not at the left boundary
                                if j != len(self.grid[i])-1: #not at the left, right boundary
                                    self.options[i][j]['right'] = 0 
                                    self.options[i][j]['left'] = 0
                                    self.options[i][j]['up'] = -1
                                    self.options[i][j]['down'] = -1 
                                else: #at the right boundary
                                    self.options[i][j]['left'] = 0
                                    self.options[i][j]['right'] = -1 
                                    self.options[i][j]['up'] = -1
                                    self.options[i][j]['down'] = -1   
                            else: #at the left boundary
                                if j != len(self.grid[i])-1: #not at the right bundary
                                    self.options[i][j]['right'] = 0
                                    self.options[i][j]['left'] = -1 
                                    self.options[i][j]['up'] = -1 
                                    self.options[i][j]['down'] = -1    
                else:
                    if i != 0: #mark upper cell
                        if self.grid[i-1][j] in valide_cells:
                            self.options[i-1][j]['down'] = -1
                    if i != len(self.grid)-1: #mark lower cell
                        if self.grid[i+1][j] in valide_cells:
                            self.options[i+1][j]['up'] = -1
                    if j != 0: #mark left cell
                        if self.grid[i][j-1] in valide_cells:
                            self.options[i][j-1]['right'] = -1
                    if j != len(self.grid[i])-1: #mark right cell
                        if self.grid[i][j+1] in valide_cells:
                            self.options[i][j+1]['left'] = -1
    
    def a_star(self):
        #neighbour {(idx1, idx2) : type}
        
        current = (0,0)
        new = (0,0)
        
        self.f_val[self.start] = self.heuristic(self.start, self.goal)
        self.g_val[self.start] = 0
        
        g_temp = 0
        f_temp = 0
        
        temp = {}
        path = {}
        
        self.p_queue.put((self.f_val[self.start], self.heuristic(self.start, self.goal), self.start))
        
        while not self.p_queue.empty():
            current = self.p_queue.get()[2]
            
            if current == self.goal:
                break
            
            directions = [k for k,v in self.options[current[0]][current[1]].items() if v == 0]
            
            for direction in directions:
                if current[0] != 0 and current[1] != 0:
                    moves = {'up': (current[0] - 1, current[1]), 'down': (current[0] + 1, current[1]), 'right': (current[0], current[1] + 1), 'left': (current[0], current[1] - 1)}
                elif current[0] == 0:
                    moves = {'up': (0, current[1]), 'down': (current[0] + 1, current[1]), 'right': (current[0], current[1] + 1), 'left': (current[0], current[1] - 1)}
                elif current[1] == 0:
                    moves = {'up': (0, current[1]), 'down': (current[0] + 1, current[1]), 'right': (current[0], current[1] + 1), 'left': (current[0], 0)}

                new = moves[direction]
                g_temp = self.g_val[current] + 1
                f_temp = self.heuristic(new, self.goal) + g_temp
                
                if new in list(self.f_val.keys()):
                    if f_temp < self.f_val[new]:
                        self.g_val[new] = g_temp
                        self.f_val[new] = f_temp
                        self.p_queue.put((self.f_val[new], self.heuristic(new, self.goal), new))
                        temp[new] = current
        
        cell = self.goal
        
        while cell != self.start:
            if cell not in temp.keys():
                print('a* failed!')
                os.execv(sys.executable, ['python'] + sys.argv) #restart program
                break
            path[temp[cell]] = cell
            cell = temp[cell]
        
        self.path = [path[k] for k in dict(reversed(list(path.items()))).keys()]
        self.path.insert(0, self.start)

    def draw_path(self, path, color):
        cell_size = self.grid_size/self.size
        
        #pygame.draw.circle(self.screen, (255, 153, 0), ((self.origin[0]+(cell_size*path[0][1] + 0.5*cell_size), self.origin[1] + (cell_size*path[0][0] + 0.5*cell_size))), 5, width=2)

        
        #start_cell:
        #Horz
        pygame.draw.line(self.screen, (255, 153, 0), (self.origin[0]+(cell_size*path[0][1]), self.origin[1] + (cell_size*path[0][0])), (self.origin[0] + (cell_size*path[0][1]+cell_size), self.origin[1] + (cell_size*path[0][0])), 2)
        pygame.draw.line(self.screen, (255, 153, 0), (self.origin[0]+(cell_size*path[0][1]), self.origin[1] + (cell_size*path[0][0]+cell_size)), (self.origin[0] + (cell_size*path[0][1]+cell_size), self.origin[1] + (cell_size*path[0][0]+cell_size)), 2)

        #Vert
        pygame.draw.line(self.screen, (255, 153, 0), (self.origin[0]+(cell_size*path[0][1]), self.origin[1] + (cell_size*path[0][0])), (self.origin[0] + (cell_size*path[0][1]), self.origin[1] + (cell_size*path[0][0]+cell_size)), 2)
        pygame.draw.line(self.screen, (255, 153, 0), (self.origin[0]+(cell_size*path[0][1]+cell_size), self.origin[1] + (cell_size*path[0][0])), (self.origin[0] + (cell_size*path[0][1]+cell_size), self.origin[1] + (cell_size*path[0][0]+cell_size)), 2)

        #goal_cell:
        #Horz
        pygame.draw.line(self.screen, (102, 0, 204), (self.origin[0]+(cell_size*path[-1][1]), self.origin[1] + (cell_size*path[-1][0])), (self.origin[0] + (cell_size*path[-1][1]+cell_size), self.origin[1] + (cell_size*path[-1][0])), 2)
        pygame.draw.line(self.screen, (102, 0, 204), (self.origin[0]+(cell_size*path[-1][1]), self.origin[1] + (cell_size*path[-1][0]+cell_size)), (self.origin[0] + (cell_size*path[-1][1]+cell_size), self.origin[1] + (cell_size*path[-1][0]+cell_size)), 2)

        #Vert
        pygame.draw.line(self.screen, (102, 0, 204), (self.origin[0]+(cell_size*path[-1][1]), self.origin[1] + (cell_size*path[-1][0])), (self.origin[0] + (cell_size*path[-1][1]), self.origin[1] + (cell_size*path[-1][0]+cell_size)), 2)
        pygame.draw.line(self.screen, (102, 0, 204), (self.origin[0]+(cell_size*path[-1][1]+cell_size), self.origin[1] + (cell_size*path[-1][0])), (self.origin[0] + (cell_size*path[-1][1]+cell_size), self.origin[1] + (cell_size*path[-1][0]+cell_size)), 2)
        
        for i in range(len(path)-1):
            """
            if self.grid[cell[0]][cell[1]] != 3:
                self.grid[cell[0]][cell[1]] = 7
            """
            
            if path[i][1] != path[i+1][1]: #step horizonally
                pygame.draw.line(self.screen, color, (self.origin[0]+(cell_size*path[i][1] + 0.5*cell_size), self.origin[1] + (cell_size*path[i][0] + 0.5*cell_size)), (self.origin[0] + (cell_size*path[i+1][1] + 0.5*cell_size), self.origin[1] + (cell_size*path[i+1][0] + 0.5*cell_size)), 2)
            
            else: #step vertically
                pygame.draw.line(self.screen, color, (self.origin[0]+(cell_size*path[i][1] + 0.5*cell_size), self.origin[1] + (cell_size*path[i][0] + 0.5*cell_size)), (self.origin[0] + (cell_size*path[i+1][1] + 0.5*cell_size), self.origin[1] + (cell_size*path[i+1][0] + 0.5*cell_size)), 2)

    def event(self, event):
        if event.type == pygame.QUIT:
            self.isRunning = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos_mouse = pygame.mouse.get_pos()
            if pos_mouse[0] > 10 and pos_mouse[0] < 10+85 and pos_mouse[1] > 475 and pos_mouse[1] < 475+25:
                grid_map = fd.askopenfilename(title = "Select the map you want to load",filetypes = (("Map files", ".csv"),))
                if grid_map:
                    self.__init__(filename = grid_map.split('/')[-1].split('.')[0], load_map = True)
                    self.init()
            if pos_mouse[0] > 150 and pos_mouse[0] < 150+102 and pos_mouse[1] > 475 and pos_mouse[1] < 475+25:
                qmap = fd.askopenfilename(title = "Select the qmap you want to load",filetypes = (("Q-Map files", ".csv"),))
                if qmap:
                    self.__init__(filename = qmap.split('/')[-1].split('.')[0], load_qmap = True)
                    self.init()

        elif event.type == pygame.KEYDOWN:
            if event.key == K_LEFT:
                self.check_status('left')

            if event.key == K_RIGHT:
                self.check_status('right')

            if event.key == K_DOWN:
                self.check_status('down')

            if event.key == K_UP:
                self.check_status('up')

            if event.key == K_s:
                self.save_map('grid')
                #self.general_txt = self.draw_general_txt(f'Saving map: {self.map_name}...')

            if event.key == K_b:
                self.stop = not self.stop
                self.action = ''
            
            if event.key == K_o:
                self.show_path = not self.show_path
            
            if event.key == K_a:
                self.show_ai_path = not self.show_ai_path
                
            if event.key == K_l:
                self.save_qmap = not self.save_qmap

    def reset(self):
        self.episode += 1
        self.actions = 0
        self.points = 0

        self.grid = copy.deepcopy(self.backup_grid)

    def check_distance(self):
        epsilon = (self.iterations+500)/(self.episode*self.epsilon+1)
        p = np.random.uniform(0.3, 1)
        idx_player = copy.deepcopy(self.start)
        idx_goal = copy.deepcopy(self.goal)
        
        for i in range(len(self.grid)):
            if 2 in self.grid[i]:
                idx_player = (self.grid[i].index(2), i)
            if 3 in self.grid[i]:
                idx_goal = (self.grid[i].index(3), i)
            

            #distance = np.sqrt((idx_player[0] - idx_goal[0])**2 + (idx_player[1] - idx_goal[1])**2) 
            distance = float(self.heuristic(idx_player, idx_goal)) # Manhattan distance
            
            if distance == 0:
                distance = 0.00001

            if (distance - self.distance_old) > 0: #distance got bigger
                self.distance_old = distance
                if p < epsilon:
                    self.dist_reward = -1*self.dist_rew*1*distance/10
            elif (distance - self.distance_old) == 0:
                self.distance_old = distance
                if p < epsilon:
                    self.dist_reward = self.dist_rew
            elif (distance - self.distance_old) < 0: #distance got smaller
                self.distance_old = distance
                if p < epsilon:
                    self.dist_reward = self.dist_rew/distance
      
    def check_status(self, command):
        visited = 5
        if self.actions > self.action_limit:
            self.reset()

        for i in range(len(self.grid)):
            if 2 in self.grid[i]:
                idx = self.grid[i].index(2)
                self.current = (idx, i)
                self.current = (i, idx)
                self.reward = self.q_map[i][idx]
                if command == 'right':
                    #check if player is at the lower border or if there is a wall above
                    if idx != (len(self.grid[i])-1) and self.grid[i][idx+1] != 1:
                        #check if player is on the win-flag
                        if self.grid[i][idx+1] == 3:
                            self.ai_path.append(self.goal)
                            self.ai_path.insert(0, self.start)
                            if len(self.ai_path) < self.steps_ai: self.steps_ai = len(self.ai_path)
                                
                            self.actions += 1
                            self.q_map[i][idx]['right'] = self.update_Q_values((i, idx+1), 3, 'right', (i, idx))
                            self.grid[i][idx] = 3
                            self.reset()
                            break
                        
                        self.ai_path.append((i,idx+1))
                        
                        self.actions += 1
                        self.q_map[i][idx]['right'] = self.update_Q_values((i, idx+1), 0, 'right', (i, idx))
                        self.grid[i][idx+1] = 2
                        self.grid[i][idx] = visited


                #check if player is at the left border or if there is a wall on the left side
                elif command == 'left':
                    if idx != 0 and self.grid[i][idx-1] != 1:
                        #check if player is on the win-flag
                        if self.grid[i][idx-1] == 3:
                            self.ai_path.append(self.goal)
                            self.ai_path.insert(0, self.start)
                            if len(self.ai_path) < self.steps_ai: self.steps_ai = len(self.ai_path)
                                
                            self.actions += 1
                            self.q_map[i][idx]['left'] = self.update_Q_values((i, idx-1), 3, 'left', (i, idx))
                            self.grid[i][idx] = 3
                            self.reset()
                            break
                        
                        self.ai_path.append((i,idx-1))
                        
                        self.actions += 1
                        self.q_map[i][idx]['left'] = self.update_Q_values((i, idx-1), 0, 'left', (i, idx))
                        self.grid[i][idx-1] = 2
                        self.grid[i][idx] = visited


                #check if player is at the lower border or if there is a wall above
                elif command == 'up':
                    if i != 0: 
                        if self.grid[i-1][idx] != 1: 
                            #check if player is on the win-flag
                            if self.grid[i-1][idx] == 3:
                                self.ai_path.append(self.goal)
                                self.ai_path.insert(0, self.start)
                                if len(self.ai_path) < self.steps_ai: self.steps_ai = len(self.ai_path)
                                    
                                self.actions += 1
                                self.q_map[i][idx]['up'] = self.update_Q_values((i-1, idx), 3, 'up', (i, idx)) 
                                self.grid[i][idx] = 3
                                self.reset()
                                break
                            
                            self.ai_path.append((i-1,idx))
                            
                            self.actions += 1
                            self.q_map[i][idx]['up'] = self.update_Q_values((i-1, idx), 0, 'up', (i, idx))
                            self.grid[i-1][idx] = 2
                            self.grid[i][idx] = visited


        #check if player is at the lower border or if there is a wall underneath
        if command == 'down':
            for i in range(len(self.grid)-1, -1, -1):
                if 2 in self.grid[i]:
                    idx = self.grid[i].index(2)
                    self.current = (i, idx)
                    if i != len(self.grid)-1:
                        if self.grid[i+1][idx] != 1:
                            #check if player is on the win-flag
                            if self.grid[i+1][idx] == 3:
                                self.ai_path.append(self.goal)
                                self.ai_path.insert(0, self.start)
                                if len(self.ai_path) < self.steps_ai: self.steps_ai = len(self.ai_path)
                            
                                self.actions += 1
                                self.q_map[i][idx]['down'] = self.update_Q_values((i+1, idx), 3, 'down', (i, idx))
                                self.grid[i][idx] = 3
                                self.reset()
                                break
                                
                            self.ai_path.append((i+1,idx))

                            self.actions += 1
                            self.q_map[i][idx]['down'] = self.update_Q_values((i+1, idx), 0, 'down', (i, idx))
                            self.grid[i+1][idx] = 2
                            self.grid[i][idx] = visited

    def draw_txt(self, txt, pos):
        self.screen.blit(txt, pos)

    def draw_general_txt(self, txt, time):
        return pygame.font.Font('freesansbold.ttf', 20).render(f'{txt}', True, (0, 0, 0))

    def draw(self):
        self.screen.fill((255,255,255))
        text = pygame.font.Font('freesansbold.ttf', 20).render(f'Episodes:  {str(self.episode)}', True, (0, 0, 0))
        text_1 = pygame.font.Font('freesansbold.ttf', 20).render(f'Actions:  {str(self.actions)}', True, (0, 0, 0))
        text_2 = pygame.font.Font('freesansbold.ttf', 20).render(f'Action:  {str(self.action)}', True, (0, 0, 0))
        text_3 = pygame.font.Font('freesansbold.ttf', 20).render(f'Reward:  {str({key : round(self.reward[key], 0) for key in self.reward})}', True, (0, 0, 0))
        text_4 = pygame.font.Font('freesansbold.ttf', 20).render(f'Mode:  {str(self.mode)}', True, (0, 0, 0))
        text_5 = pygame.font.Font('freesansbold.ttf', 20).render(f'Distance:  {str(round(self.distance_old, 0))}', True, (0, 0, 0))
        text_6 = pygame.font.Font('freesansbold.ttf', 20).render(f'Save-Qmap:  {str(self.save_qmap)}', True, (0, 0, 0))
        text_7 = pygame.font.Font('freesansbold.ttf', 20).render(f'Map_name:  {str(self.map_name)}', True, (0, 0, 0))
        text_8 = pygame.font.Font('freesansbold.ttf', 20).render(f'Steps a*:  {str(len(self.path))}', True, (0, 0, 0))
        text_9 = pygame.font.Font('freesansbold.ttf', 20).render(f'Steps AI:  {str(self.steps_ai)}', True, (0, 0, 0))
        text_10 = pygame.font.Font('freesansbold.ttf', 20).render(f'Commands:', True, (100, 100, 0))
        text_11 = pygame.font.Font('freesansbold.ttf', 20).render(f'S - Save map', True, (75, 75, 75))
        text_12 = pygame.font.Font('freesansbold.ttf', 20).render(f'L - Save qmap', True, (75, 75, 75))
        text_13 = pygame.font.Font('freesansbold.ttf', 20).render(f'O - toggle a* visualization', True, (75, 75, 75))
        text_14 = pygame.font.Font('freesansbold.ttf', 20).render(f'B - toggle learning', True, (75, 75, 75))
        #text_10 = self.general_txt

        self.draw_txt(text, (10, 25))
        self.draw_txt(text_1, (10, 50))
        self.draw_txt(text_2, (10, 75))
        self.draw_txt(text_3, (200, 25))
        self.draw_txt(text_4, (10, 125))
        self.draw_txt(text_5, (10, 100))
        self.draw_txt(text_6, (10, 150))
        self.draw_txt(text_8, (10, 200))
        self.draw_txt(text_9, (10, 225))
        self.draw_txt(text_10, (10, 300))
        self.draw_txt(text_11, (10, 325))
        self.draw_txt(text_12, (10, 350))
        self.draw_txt(text_13, (10, 375))
        self.draw_txt(text_14, (10, 400))

        pygame.draw.rect(self.screen, (0,0,0), (10, 475, 85, 25),2)
        pygame.draw.rect(self.screen, (100,0,0), (12, 477, 82, 22))
        btn_load_txt = pygame.font.Font('freesansbold.ttf', 16).render(f'Load map', True, (0, 0, 0))
        self.draw_txt(btn_load_txt, (15, 480))

        pygame.draw.rect(self.screen, (0,0,0), (150, 475, 102, 25),2)
        pygame.draw.rect(self.screen, (100,0,0), (152, 477, 99, 22))
        btn_load_txt = pygame.font.Font('freesansbold.ttf', 16).render(f'Load Q-Map', True, (0, 0, 0))
        self.draw_txt(btn_load_txt, (155, 480))
        
        if self.map_name:
            self.draw_txt(text_7, (10, 175))

        self.drawGrid(self.size)
        self.placeObjects()
        
        if self.show_path:
            self.draw_path(self.path, (50,100,50))
            
        if self.episode > self.check_epsiode or self.show_ai_path:
            self.draw_path(self.ai_path, (100,0,0))

        pygame.display.update()

    def execute(self):
        if self.init() == False:
            self.isRunning = False

        while self.isRunning and self.episode < self.episode_limit:
            for event in pygame.event.get():
                self.event(event)

            if not self.stop:
                self.learn()
            
            if self.episode == self.check_epsiode:
                self.copy_path = True
            else:
                self.copy_path = False
            
            self.check_status(self.action)
            self.check_distance()
            self.root.withdraw()
            self.draw()
            self.clock.tick(self.fps)
        
        if len(self.path) == len(self.ai_path):
            print('Optimal Path found!')
        
        if self.save_qmap:
            self.save_map("qmap")
        pygame.quit()

if __name__ == "__main__":
    np.seterr(over='raise')
    p = Simulation()
    p.execute()