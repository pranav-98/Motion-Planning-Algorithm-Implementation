# Basic searching algorithms

# Class for each node in the grid
class Node:
    def __init__(self, row, col, is_obs, h):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.is_obs = is_obs  # obstacle?
        self.g = None         # cost to come (previous g + moving cost)
        self.h = h            # heuristic
        self.cost = None      # total cost (depend on the algorithm)
        self.parent = None    # previous node


#This funtion is used to return the path 
def parent_trace(path, finalNode):
    rp=finalNode
    while(rp.parent):
        
                    
        path=[[rp.parent.row,rp.parent.col]]+path
                                            
        rp=rp.parent
    return path
                    
        


def bfs(grid, start, goal):
    '''Return a path found by BFS alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> bfs_path, bfs_steps = bfs(grid, start, goal)
    It takes 10 steps to find a path using BFS
    >>> bfs_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False
    q=[]
    visited=[]
    
    
    [a,b]=start
    beg=Node(a,b,grid[a][b],h=0)
    #Appending starting node to the list q
    
    q.append(beg)
    
    coor=[]

    
    rows=[0,1,0,-1]
    cols=[1,0,-1,0]
    
    while q:
    #While the queue is not empty these operations will follow    
        
        prevNode=q.pop(0)
        #Popping the first element
        steps+=1
        for k in q:
            coor.append([k.row,k.col])
        #Keeping track of nodes in the queue
        
        present=[prevNode.row,prevNode.col]


        visited.append(present)
        #Keeping track of visited nodes


        for i in range(0,len(rows)):
            #Visiting neighbours and eliminating based on conditions
            nr= present[0]+rows[i]
            nc= present[1]+cols[i]

            new=[nr,nc]
            

            if nr<0 or nc<0:
                continue
               
               
            if nr>=len(grid) or nc>=len(grid[0]):
                continue
           
           
            if grid[nr][nc]==1:
                continue

            if new not in visited and new not in coor :
                newNode=Node(nr,nc,grid[nr][nc],h=0)
                newNode.parent=prevNode
                q.append(newNode)
                
        
        
        if [prevNode.row,prevNode.col]==goal:
                newNode=Node(prevNode.row,prevNode.col,grid[prevNode.row][prevNode.col],h=0)
                newNode.parent=prevNode
                q.append(newNode)
                found=True
                path=parent_trace(path,newNode)
                
                    
                break                                   
                          

    if found:
        print(f"It takes {steps} steps to find a path using BFS")
    else:
        print("No path found")
    return path, steps


def dfs(grid, start, goal):
    '''Return a path found by DFS alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dfs_path, dfs_steps = dfs(grid, start, goal)
    It takes 9 steps to find a path using DFS
    >>> dfs_path
    [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    q=[]
    visited=[]
    
    
    [a,b]=start
    beg=Node(a,b,grid[a][b],h=0)
    #Creating the first Node
    
    q.append(beg)
    #Appending the first node to the list
    
    
    rows=[-1,0,1,0]
    cols=[0,-1,0,1]
    while q:
        
        
        prevNode=q.pop(-1)
        steps+=1
        #Popping the last element in the queue

           
        present=[prevNode.row,prevNode.col]
        visited.append(present)
        #Keeping track of visited nodes   
        
        for i in range(0,len(rows)):
            #Going through the neighbours
            nr= present[0]+rows[i]
            nc= present[1]+cols[i]

            new=[nr,nc]
            

            
   
            if nr<0 or nc<0:
                continue
               
               
            if nr>=len(grid) or nc>=len(grid[0]):
                continue
           
           
            if grid[nr][nc]==1:
                continue

            

            
            if new not in visited :
                newNode=Node(nr,nc,grid[nr][nc],h=0)
                newNode.parent=prevNode
                q.append(newNode)
                #Creating a new node and appending it to the queue
                
        
        
        if [prevNode.row,prevNode.col]==goal:
                

                newNode=Node(prevNode.row,prevNode.col,grid[prevNode.row][prevNode.col],h=0)
                newNode.parent=prevNode
                q.append(newNode)
                found=True
                path=parent_trace(path,newNode)
                break
    
    if found:
        print(f"It takes {steps} steps to find a path using DFS")
    else:
        print("No path found")
    return path, steps


def dijkstra(grid, start, goal):
    '''Return a path found by Dijkstra alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dij_path, dij_steps = dijkstra(grid, start, goal)
    It takes 10 steps to find a path using Dijkstra
    >>> dij_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    q=[]
    visited=[]
    
    
    [a,b]=start
    beg=Node(a,b,grid[a][b],h=0)
    beg.cost=0
    
    q.append(beg)
    
    
    
    
    rows=[0,1,0,-1]
    cols=[1,0,-1,0]
    while q:
        
        q.sort(key=lambda x:x.cost)
        prevNode=q.pop(0)
        steps+=1
        
        
        present=[prevNode.row,prevNode.col]
        visited.append(present)
        
           
        
        for i in range(0,len(rows)):
            t=0
            nr= present[0]+rows[i]
            nc= present[1]+cols[i]

            new=[nr,nc]
            

            
   
            if nr<0 or nc<0:
                continue
               
               
            if nr>=len(grid) or nc>=len(grid[0]):
                continue
           
           
            if grid[nr][nc]==1:
                continue

            
            
            
            if new not in visited :
                
                
                for k in q:
                    if k.row==new[0] and k.col==new[1]:
                        t=1 
                        k.cost=min(k.cost,prevNode.cost+1)
                    

                if t==0:
                    newNode=Node(nr,nc,grid[nr][nc],h=0)
                    newNode.cost=prevNode.cost+1
                    newNode.parent=prevNode
                    q.append(newNode)
                    

        
        
        if present==goal:
            newNode=Node(prevNode.row,prevNode.col,grid[prevNode.row][prevNode.col],h=0)
            newNode.parent=prevNode
            q.append(newNode)
            
            found=True
            path=parent_trace(path,newNode)
                    
            break
    
    
    if found:
        print(f"It takes {steps} steps to find a path using Dijkstra")
    else:
        print("No path found")
    return path, steps

def manh(a,b,c,d):
    k=((abs(c-a))+(abs(d-b)))
    return k

def astar(grid, start, goal):
    '''Return a path found by A* alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> astar_path, astar_steps = astar(grid, start, goal)
    It takes 7 steps to find a path using A*
    >>> astar_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False


    q=[]
    visited=[]
    
    
    [a,b]=start
    beg=Node(a,b,grid[a][b],h=manh(a,b,goal[0],goal[1]))
    beg.cost=0
    
    q.append(beg)
    
    coor=[]
    
    
    rows=[0,1,0,-1]
    cols=[1,0,-1,0]
    while q:
        
        
        q.sort(key=lambda x:(x.h+x.cost))
        prevNode=q.pop(0)
        steps+=1


        present=[prevNode.row,prevNode.col]
        visited.append(present)
        
           
        for i in range(0,len(rows)):
            t=0
            nr= present[0]+rows[i]
            nc= present[1]+cols[i]

            new=[nr,nc]
            

            
   
            if nr<0 or nc<0:
                continue
               
               
            if nr>=len(grid) or nc>=len(grid[0]):
                continue
           
           
            if grid[nr][nc]==1:
                continue

            
            
            
            if new not in visited :
                
                for k in q:
                    if k.row==new[0] and k.col==new[1]:
                        t=1 
                        k.cost=min(k.cost,prevNode.cost+1)
                    

                if t==0:
                    newNode=Node(nr,nc,grid[nr][nc],h=manh(nr,nc,goal[0],goal[1]))
                    newNode.cost=prevNode.cost+1
                    newNode.parent=prevNode

                    
                    q.append(newNode)
        
        
        if present==goal:
            
            newNode=Node(prevNode.row,prevNode.col,grid[prevNode.row][prevNode.col],h=0)
            newNode.parent=prevNode
            q.append(newNode)
            found=True
            path=parent_trace(path,newNode)
                    
            break

    
    if found:
        print(f"It takes {steps} steps to find a path using A*")
    else:
        print("No path found")
    return path, steps


# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()
