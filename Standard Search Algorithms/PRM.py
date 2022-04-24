# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree
import random



# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path
        self.Kdtree=[]

    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        ### YOUR CODE HERE ###
        lineeqn=zip(np.linspace(p1[0],p2[0],dtype=int), np.linspace(p1[1],p2[1],dtype=int))
        for l in lineeqn:
            if self.map_array[l[0]][l[1]]==0:
                return True
        return False


    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        euclidean=((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5
        return euclidean


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        xpts=np.linspace(0, self.size_row-1, int(n_pts**0.5),dtype=int)
        ypts=np.linspace(0, self.size_col-1, 25,dtype=int)
        
        
        
        for x in xpts:
            for y in ypts:
                
                if self.map_array[x][y]==1 and (x,y) not in self.samples:
                    self.samples.append((x,y))
                    
            
        return self.samples

    
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        for i in range(0,n_pts):
            xpts= random.randint(0, self.size_row-1)
            ypts=random.randint(0, self.size_col-1)

            if self.map_array[xpts][ypts]==1 and (xpts,ypts) not in self.samples:
                self.samples.append((xpts,ypts))
        
        return self.samples


    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        for i in range(0, n_pts):
            x=random.randint(0, self.size_row-1)
            y=random.randint(0, self.size_col-1)

            xpt=int(np.random.normal(loc=x,scale=15))
            xpt=min(xpt,self.size_row-1)
            xpt=max(0, xpt)
            ypt=int(np.random.normal(loc=y,scale=10))
            ypt=min(ypt,self.size_col-1)
            ypt=max(0,ypt)
            if self.map_array[xpt][ypt]==1 and self.map_array[x][y]==0 and (xpt,ypt) not in self.samples:
                self.samples.append((xpt,ypt))

            elif self.map_array[x][y]==1 and self.map_array[xpt][ypt]==0 and (x,y) not in self.samples:
                self.samples.append((x,y))

            else:
                continue
            
        return self.samples


    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        obs=[]
        #for i in range(0, n_pts):
        #    x1=np.random.randint(0, self.size_row-1,size=2)
        #    y1=np.random.randint(0,self.size_col-1,size=2)
        #    midx=int((x1[0]+x1[1])/2)
        #    midy=int((y1[0]+y1[1])/2)

        #    if self.map_array[x1[0]][y1[0]]==0 and self.map_array[x1[1]][y1[1]]==0 and (midx,midy) not in self.samples:
        #        if self.map_array[midx][midy]==1:
        #            self.samples.append((midx,midy))

        for i in range(0, n_pts):
            x1=np.random.randint(0, self.size_row-1)
            y1=np.random.randint(0,self.size_col-1)
            if self.map_array[x1][y1]==0:
                # x2=np.random.randint(0, self.size_row-1)
                # y2=np.random.randint(0,self.size_col-1)
                x2=max(int(np.random.normal(loc=x1,scale=20)),0)
                x2=min(int(np.random.normal(loc=x1,scale=20)),self.size_row-1)
                y2=max(int(np.random.normal(loc=y1,scale=20)),0)
                y2=min(int(np.random.normal(loc=y1,scale=20)),self.size_col-1)

                if self.map_array[x2][y2]==0:
                    x=int((x1+x2)/2)
                    y=int((y1+y2)/2)
                    if self.map_array[x][y]==1 and (x,y) not in self.samples:
                        self.samples.append((x,y))


            
        return self.samples


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02), 
        #          (p_id1, p_id2, weight_12) ...]
        

        
        pairs = []

        
        self.Kdtree = KDTree(self.samples)              
        
        index1=0
        for i in self.samples:
            

            _,ind=self.Kdtree.query([i], k=10,)
            
            for index in ind[0]:
                


                j=self.samples[index]
                if i==j:
                    continue
                
                if self.check_collision(i,j)==False:
                    pairs.append([index1,index,self.dis(i,j)])
            index1+=1
        
        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01), 
        #                                     (p_id0, p_id2, weight_02), 
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of 
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from([])
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))


    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        ### YOUR CODE HERE ###
        
        start_coor=self.samples[-2]
        goal_coor=self.samples[-1]

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1), 
        #                (start_id, p_id2, weight_s2) ...]
        start_pairs = []
        goal_pairs = []

        _,start_index=self.Kdtree.query([start_coor],k=10)
        for i in start_index[0]:
            j=self.samples[i]
            if start_coor==j:
                continue

            if self.check_collision(start_coor, j)==False:
                start_pairs.append(['start',i,self.dis(start_coor,j)])


        _,goal_index=self.Kdtree.query([goal_coor],k=10)
        for i in goal_index[0]:
            j=self.samples[i]
            if goal_coor==j:
                continue

            if self.check_collision(goal_coor, j)==False:
                goal_pairs.append(['goal', i,self.dis(goal_coor,j)])


        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        