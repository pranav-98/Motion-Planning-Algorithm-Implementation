# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

goal_bias=5
# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag
        

    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    
    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        eud= ((node1.row-node2.row)**2+(node1.col-node2.col)**2)**0.5
        return eud

    
    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        lineeqnr=np.linspace(node1.row,node2.row,10,dtype=int)
        lineeqnc=np.linspace(node1.col,node2.col,10,dtype=int)
        
        
        line=zip(lineeqnr,lineeqnc)
        #print(line)

        for l in line:
            
            if self.map_array[l[0]][l[1]]==0:
                return True

        return False




    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###

        t=np.random.randint(0,100)
        if t<=goal_bias:
            return [self.goal.row,self.goal.col]

        rpt=np.random.randint(0,self.size_row-1)
        cpt=np.random.randint(0,self.size_col-1)

        point=[rpt,cpt]

        return point
        


    
    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###return
        treevar=[]
        
        
        if len(self.vertices)==1:
            return self.vertices[0]

        for no in self.vertices:
            treevar.append((no.row,no.col))

        #treevar.append((point[0],point[1]))
        tree=KDTree(treevar)

        _,Nn=tree.query([[point[0],point[1]]],k=2)
        
        pointNn=self.vertices[Nn[0][1]]
        

        return pointNn
                

        


    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance 
        '''
        ### YOUR CODE HERE ###
        treeval=[]
        for v in self.vertices:
            treeval.append((v.row,v.col))

        #treeval.append((new_node.row,new_node.col))
        neighbour=[]
        tree=KDTree(treeval)
        

        ind=tree.query_radius([[new_node.row,new_node.col]], r=neighbor_size,return_distance=False,count_only=False)
        
        for i in ind[0]:
            
            neighbour.append(self.vertices[i])
            
        
        return neighbour




    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        
        NODE=neighbors[0]
        min_cost=neighbors[0].cost+self.dis(new_node, neighbors[0])
        ### YOUR CODE HERE ###
        for no in neighbors:
            cost=no.cost+self.dis(no,new_node)
            if cost<min_cost:
                new_node.cost=cost
                new_node.parent=no
                NODE=no

        neighbors.remove(NODE)

        for no in neighbors:
            if no.cost>(new_node.cost+self.dis(new_node, no)):
                no.parent=new_node
                no.cost= (new_node.cost+self.dis(new_node, no))
        



        

    
    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')
        
        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col and cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()


    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.
        self.start.parent=self.start
        self.start.cost=0
        for i in range(0, n_pts):
            point=self.get_new_point(goal_bias)
            if self.map_array[point[0]][point[1]]==0:
                continue
            Nn=self.get_nearest_node(point)
            
            # if self.check_collision(NewNode, Nn)== True:
            #     continue
            
            # self.vertices.append(NewNode)
            

            
            slope=np.arctan2(-1*(Nn.col-point[1]),-1*(Nn.row-point[0]))
                
            point[0]=Nn.row+10*np.cos(slope)
            point[1]=Nn.col+10*np.sin(slope)
            
            
            if self.map_array[int(point[0])][int(point[1])]==0:
                continue
            NewNode=Node(point[0],point[1])
            
            if self.check_collision(Nn,NewNode)==False:
                
                NewNode.parent=Nn
                NewNode.cost=Nn.cost+self.dis(NewNode,Nn)
                
                self.vertices.append(NewNode)
                
            else:
                continue

            if self.dis(NewNode,self.goal)<5:
                
                self.found=True
                
                self.goal.parent=NewNode
                self.goal.cost=NewNode.cost+self.dis(NewNode,self.goal)
                self.vertices.append(self.goal)
                break
            
            

        
        

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
        # Draw result
        self.draw_map()


    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            neighbor_size - the neighbor distance
        
        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.
        self.start.parent=self.start
        self.start.cost=0
        for i in range(0, n_pts):
            point=self.get_new_point(goal_bias)
            if self.map_array[point[0]][point[1]]==0:
                continue
            Nn=self.get_nearest_node(point)
            
            # if self.check_collision(NewNode, Nn)== True:
            #     continue
            
            # self.vertices.append(NewNode)
            

            
            slope=np.arctan2(-1*(Nn.col-point[1]),-1*(Nn.row-point[0]))
                
            point[0]=Nn.row+10*np.cos(slope)
            point[1]=Nn.col+10*np.sin(slope)

            point[0]=min(point[0],self.size_row-1)
            point[0]=max(point[0],0)
            point[1]=min(point[1],self.size_col-1)
            point[1]=max(point[1],0)
            
            if self.map_array[int(point[0])][int(point[1])]==0:
                continue
            NewNode=Node(point[0],point[1])
            
            if self.check_collision(Nn,NewNode)==False:
                
                NewNode.parent=Nn
                NewNode.cost=Nn.cost+self.dis(NewNode,Nn)
                
                neighbour=self.get_neighbors(NewNode, neighbor_size=20)
                if len(neighbour)==0:
                    continue
                self.rewire(NewNode,neighbour)
                
                self.vertices.append(NewNode)
                # print("Nearest Neighbour",[Nn.row,Nn.col])
                # print("New Node",[point[0],point[1]])
            else:
                continue

            if self.dis(NewNode,self.goal)<5:
                
                self.found=True
                
                self.goal.parent=NewNode
                self.goal.cost=NewNode.cost+self.dis(NewNode,self.goal)
                neighbour1=self.get_neighbors(self.goal, neighbor_size=10)
                if len(neighbour1)==0:
                    continue
                self.rewire(self.goal,neighbour1)
                self.vertices.append(self.goal)
                

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
