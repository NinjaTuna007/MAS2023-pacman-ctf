# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import numpy as np
from typing import List, Tuple
from util import nearestPoint
from baselineTeam import OffensiveReflexAgent

#################
# Team creation #
#################


# def createTeam(firstIndex, secondIndex, isRed,
#                first = 'DummyAgent', second = 'DummyAgent'):
def createTeam(firstIndex, secondIndex, isRed,
               first = 'HybridAgent', second = 'HybridAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

    # bool: if on read team then true, if false then player is on blue team
    self.teamRed = gameState.isOnRedTeam(self.index) 
    #self.walls = gameState.data.layout.walls
    self.walls = gameState.getWalls()

    self.defendFood = CaptureAgent.getFoodYouAreDefending(self, gameState)
    






  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    # ogPos = gameState.getAgentPosition(self.index)
    # print("before current "+ str(ogPos) )

    # doAbleActions = []
    # for action in actions:
    #   print("action " + str(action))

    #   successor = self.getSuccessor(gameState, action) # getSuccessor is from class Action
    #   print("successor: "+ str( type(successor)) + " self "+ str(type(self) ))
    

    #   if successor.getAgentState(self.index).getPosition() != gameState.getAgentPosition(self.index):
    #     doAbleActions.append(action)
    #     print("this happened")
    #   else:
    #     print("[id] "+ str(self.index) + " current "+ str(ogPos) + " successor "+ str(successor.getAgentState(self.index).getPosition()))

    # actions = doAbleActions
    

    return random.choice(actions)
  


  # successor if new position after executing 'action'
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """

    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    


  def GetFriendsNFoes(self, successor: game.AgentState)->List[List[int]]:
    """get indexes of [friends, foes]"""
    friends = [successor.getAgentState(i) for i in self.getTeam(successor)]
    foes = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    return [friends, foes]
  
  def getAgentPosition(self, successor: game.AgentState, indx: int)-> Tuple:
    """given an index of an agent, returns Tuple of position in grid"""
    return successor.getAgentState(indx)

  
  
    













  
  def isWall(self, position: tuple):
    """
    Check if position is a tuple
    """
    return self.walls[position[0]][position[1]]
  
  def FindEnemyPosFromFood(self, gameState)->List[Tuple]:
    """ if food count decreases since last call => enemy is there """

    if self.teamRed:
      myFood = gameState.getRedFood()
    else: 
      myFood = gameState.getBlueFood()
    
    enemyPos = None
    if myFood.count() != self.defendFood.count(): # if our food has decreased:
      # go through all food, find which position is missing food
      enemyPos = [(i,k) for i in range(self.defendFood.width) for k in range(self.defendFood.height) 
                  if self.defendFood[i][k] != myFood[i][k]]
    self.defendFood = myFood
    
    return enemyPos

      
  # def GetState(self,gameState):
  #   return gameState.getSuccessor()

























class HybridAgent( OffensiveReflexAgent):
    """Hybridagent is both offensive and defensive depending on situation"""
    

    def registerInitialState(self, gameState):
        startTime_registerInitialState = time.time()
        """registering initial state, command is required to run game"""
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)

        # bool: if on read team then true, if false then player is on blue team
        self.teamRed = gameState.isOnRedTeam(self.index) 

        
        
        # bool for pacman to return home
        self.PacRetrun = False
        
        # get the game walls
        self.walls = gameState.getWalls()

        # food i'm defending, in form of a matrix where m[x][y]=true if there is food at (x,y) that your opponent can eat
        self.defendFood = CaptureAgent.getFoodYouAreDefending(self, gameState)

        # memory matrix of where food was before enemy attack, to be used to find out where enemies are
        self.foodBeforeAttack = CaptureAgent.getFoodYouAreDefending(self, gameState)

        # startTime = time.time()
        initialFoodCount = GetFoodCount(self.defendFood)
        # elapsedTime = time.time() - startTime
        self.originalNumberOfFood = initialFoodCount
        self.foodCountBeforeEnemyAttack = initialFoodCount
        
        # print("number of foodcapsules are "+ str(self.originalNumberOfFood))

        # initial roles setup in start, they start at different positions, set first one to defense,second to offense

        
        self.canEatCount = self.GetEatCount(gameState)
        # print(self.canEatCount)
        self.foodThreshold = int( (self.canEatCount-2)/5 )

        # print initial position of agents
        # print("[id]"+str(self.index)+"initial position of agents"+ str(gameState.getAgentPosition(self.index)))
        
        # assign roles, no particular reason for this order, they are essentially in same position
        self.AssignRoles(gameState)

        # print("[time] registerInitialState: "+ str(time.time()-startTime_registerInitialState))


    
    # function that assigns roles to agents
    def AssignRoles(self, gameState):
        """assigns roles to agents"""
        if self.index == self.getTeam(gameState)[0]:
            self.role = "defense"
        else:
            self.role = "offense"
        # print("agent "+ str(self.index) + " role is " + self.role)
        pass


    def FindClosest2Eat(self, gameState):
        """finds closest food or capsule to me to aim towards"""
        capsules = self.getCapsules(gameState) # list of tuples
        food = self.getFood(gameState) #grid where m[x][y]=true if there is food at (x,y)

        closestCapsuleDist = np.inf
        closestCapsule = None
        # check so capsule list is not empty
        if len(capsules) > 0:
          # check all capsule tuples in list of capsules, save shortest distance to myPos
          for capsule in capsules:
            capsuleDist = self.getMazeDistance(gameState.getAgentPosition(self.index), capsule)
            if capsuleDist < closestCapsuleDist:
              closestCapsuleDist = capsuleDist
              closestCapsule = capsule
              # closestCapsule = min( closestCapsule, self.getMazeDistance(gameState.getAgentPosition(self.index), capsule) )


        # print("food"+ str(food))
        # print("  ")
        # print("capsules"+ str(capsules))

        
        # if food is closer than capsule, then return food, else return capsule
        
        # loop through food, remember closest food
        closestFood = self.FindClosestFood(gameState, food)
        # print("closest food is "+ str(closestFood))
        
        # print("closest capsule is "+ str(closestCapsule))

        myPos = gameState.getAgentPosition(self.index)

        if closestCapsule != None and closestFood != myPos: # there exist a capsule, and there is still food to eat
          if self.getMazeDistance(myPos, closestFood) < self.getMazeDistance(myPos, closestCapsule):
              return closestFood
          else:
              return closestCapsule
        elif closestCapsule == None and closestFood != myPos: # there is no capsule, but there is still food to eat
          return closestFood  
        else: return None # there is no food to eat, return None
           
    
    def FindClosestFood(self, gameState, matrix:game.Grid)->Tuple[int]:
        """finds closest food and returns it's tuple point"""
        myPos = gameState.getAgentPosition(self.index)
        
        minDist = np.inf

        # tupleMemory 
        tupleMemory = myPos
        rowCount = 0

        for row in matrix :
            colCount = 0
            # print("row: "+ str(row))

            for element in row:
                # print("rowcount "+ str(rowCount) + " colcount "+ str(colCount) + " element "+ str(element))

                try:
                  if element: # this is food
                      currDist = self.getMazeDistance(myPos, (rowCount,colCount))
                      if currDist < minDist:
                          minDist = currDist
                          tupleMemory = (rowCount,colCount) #(myPos, element)
                except:
                  # print("error in FindClosestFood")
                  pass

                
                colCount += 1
            rowCount+= 1
        return tupleMemory
    
    def GetEatCount(self, gameState)->int:
        """counts food and capsules that can eat"""
        capsules = self.getCapsules(gameState)
        food = self.getFood(gameState)

        numbOfFood = GetFoodCount(food)
        numbOfCapsules = GetFoodCount(capsules)
        return numbOfFood+ numbOfCapsules

    def getFeatures(self, gameState, action):
      
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)

        #--- info of my state----
        
        # my new state/position after executing given action
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        eatenFood = gameState.getAgentState(self.index).numCarrying

        # enemy capsules want to eat
        enemyCapsules = self.getCapsules(gameState)
        enemyCapsulesInt = len(enemyCapsules)
        #-------------

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]


        # check if i am ghost and enemy is pacman and is near me
        if not myState.isPacman and myState.scaredTimer == 0 and len(enemies) > 0 and self.role == "offense":
           features = self.AssignDefensiveFeatures(gameState, enemies, myPos, features)
        
        # I am offensive agent, no pacman near me, i go eat food
        elif self.role == "offense": 
           features = self.AssignOffencsiveFeatures(gameState, enemies,myPos, features)
        
        # I am defensive agent
        else: 
            # debug draw on myPos
            self.debugDraw(myPos, [0,1,0], clear=True) # green dot
            features =  self.AssignDefensiveFeatures(gameState, enemies, myPos, features)

        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        

        # check if stop or reverse
        if action == Directions.STOP: features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        if action == rev: features['reverse'] = 1  

        # check how much food each enemy has eaten
        features['foodEaten'] = 0
        for enemy in invaders:
            enemy_food_carried = enemy.numCarrying
            features['foodEaten'] = max(enemy_food_carried, features['foodEaten'])

        return features



    def AssignOffencsiveFeatures(self, gameState, enemies, myPos, features):
      """assigns features for offensive agent, i am pac, running away from ghosts"""
      # check if I am pacman or ghost
      features['onDefence'] = 0
      #check if enemy is ghost, if ghost then check distance to enemy
      ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
      
      amPac = gameState.getAgentState(self.index).isPacman
      # todo: change so that checks if am ghost then don't run from ghosts

      if len(dists) > 0:
          # check if ghost is scared
          if gameState.getAgentState(self.index).scaredTimer > 0:
              # print("scared ghost")
              features['exactEnemyDistanceScaredGhost'] = min(dists)
          else: # not scared
              features['exactEnemyDistance'] = min(dists)

      else: # use fuzzy distance 
        # get noisy enemy distance
        noisyDistance = gameState.getAgentDistances()
        # keep only enemy distances
        noisyDistance = [noisyDistance[i] for i in self.getOpponents(gameState)]      
        # get closest enemy
        closestEnemy = min(noisyDistance)
        if gameState.getAgentState(self.index).scaredTimer > 0:
          features['noisyEnemyDistanceScaredGhost'] = closestEnemy
        else: # not scared
          features['noisyEnemyDistance'] = closestEnemy

      # get closest food or capsule
      closestFood = self.FindClosest2Eat(gameState)
      features['distance2food'] = self.getMazeDistance(myPos, closestFood)
      
      
      #use debugDraw to draw red around offensive agent
      self.debugDraw(myPos, [1,0,0], clear=True)

      return features

    def AssignDefensiveFeatures(self, gameState, enemies, myPos, features):
       """get features for defensive agent"""
       features['onDefence'] = 1
    
       #check if enemy is pacman, if pacman then check distance to enemy
       pacs = [a for a in enemies if a.isPacman and a.getPosition() != None]
       dists = [self.getMazeDistance(myPos, a.getPosition()) for a in pacs]
       if len(dists) > 0:
         # check if i am not scared and that enemy is not one step away from half of the map
         # get half of the map
         halfMap = gameState.data.layout.width/2
         # get enemy position
         enemyPos = pacs[0].getPosition() 
         if gameState.getAgentState(self.index).scaredTimer == 0 and enemyPos[0] < halfMap - 1:
             features['exactEnemyDistance'] = min(dists)
         else:
             features['exactEnemyDistanceScaredGhost'] = min(dists) 
       
         invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
         features['numberOfInvaders'] = len(invaders)
         if len(invaders) > 0:
             dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
             # check if ghost is scared
             if gameState.getAgentState(self.index).scaredTimer > 0:
                 features['exactEnemyDistanceScaredGhost'] = 1
             else:
                 features['exactEnemyDistance'] = min(dists)
         else: # use fuzzy distance
             # get noisy enemy distance  
             noisyDistance = gameState.getAgentDistances()
             # keep only enemy distances
             noisyDistance = [noisyDistance[i] for i in self.getOpponents(gameState)]
             # get closest enemy
             closestEnemy = min(noisyDistance)
             if gameState.getAgentState(self.index).scaredTimer > 0: # if scared
                 features['noisyEnemyDistanceScaredGhost'] = closestEnemy
             else: # not scared
                 features['noisyEnemyDistance'] = closestEnemy
       return features


    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)

        # weights is getWeightsOffensive if features['onDefence'] == 0 , getWeightsDefensive if features['onDefence'] == 1 and getWeightsReturnHome if above threshold
        # if features['onDefence'] == 0 and features['foodEaten'] < self.foodThreshold:
        #     weights = self.getWeightsOffensive()
        # elif features['onDefence'] == 1 :
        #     weights = self.getWeightsDefensive()
        # else:
        #     weights = self.getWeightsReturnHome()

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]


        features = self.getFeatures(gameState, action)
        # weights = self.getWeights(gameState, action)

        # check if i am ghost and enemy is pacman and is near me
        if not myState.isPacman and myState.scaredTimer == 0 and len(enemies) > 0 and self.role == "offense":
          weights = self.getWeightsDefensive()
        # I am offensive agent, no pacman near me, i go eat food
        elif self.role == "offense": 
          weights = self.getWeightsOffensive()
        else:
          weights = self.getWeightsDefensive()






        # if pacman check closest food, penalize being away from it
        # am i pacman?
        # isPac = gameState.getAgentState(self.index).isPacman
        # if isPac:
        #     # find closest food, penalize being away from it
        #     closestFood = self.FindClosest2Eat(gameState)
        #     myPos = gameState.getAgentPosition(self.index)
        #     features['distance2food'] = self.getMazeDistance(myPos, closestFood)


        # call on CheckIfEnemyInMyTerritory, if true then call on findenemyinmyterritory, penalize going away from enemy
        # if self.CheckIfEnemyInMyTerritory(gameState):
        #     # find enemy in my territory, penalize going away from enemy
        #     enemyPos = self.FindEnemyInMyTerritory(gameState)
        #     if(enemyPos != None):
        #         # my position is my current position
        #         myPos = gameState.getAgentPosition(self.index)
        #         print("enemy pos is "+ str(enemyPos))
        #         print("my pos is "+ str(myPos))
        #         features['exactEnemyDistance'] = self.getMazeDistance(myPos, enemyPos)


        return features * weights

    def CheckIfEnemyInMyTerritory(self, gameState)->bool:
        """checks if enemy is in my territory"""
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            return True
        else:
           #check if old food count is same as current food count
            currentFoodCount = GetFoodCount(CaptureAgent.getFoodYouAreDefending(self, gameState))
            if currentFoodCount < self.foodCountBeforeEnemyAttack:
                # self.foodCountBeforeEnemyAttack = currentFoodCount
                return True
            else:
                return False

    def FindEnemyInMyTerritory(self, gameState)->Tuple[int]:
        """finds enemy in my territory by comparing old food count with current food count"""
        # todo: check why this function is not working
       # check if old food count is same as current food count
       # if different check where food is missing
        currentFoodCount = GetFoodCount(CaptureAgent.getFoodYouAreDefending(self, gameState))
        if currentFoodCount < self.foodCountBeforeEnemyAttack:
            # print("food count before enemy attack is "+ str(self.foodCountBeforeEnemyAttack) + " current food count is "+ str(currentFoodCount))
            self.foodCountBeforeEnemyAttack = currentFoodCount
            # find where food is missing

            print("self.defendfood dimensions are "+ str(self.defendFood.height) + " " + str(self.defendFood.width))

            for row in range(self.defendFood.height):
                for col in range(self.defendFood.width):
                    # print("row "+ str(row) + " col "+ str(col))
                    # print("self defendfood "+ str(self.defendFood[row][col]))
                    # print(str(gameState.hasFood(row, col)))
                    if self.defendFood[row][col] and not gameState.hasFood(row, col):
                        self.defendFood = gameState.getAgentState(self.index).getFoodYouAreDefending()
                        
                        return tuple(row,col)

    def getWeightsDefensive(self):
        """interpetation:
            numberOfInvaders: punish for number of invaders
            onDefense: give reward for defending home
            stop: punish executing the action "stop"
            fuzzyEnemyDistance: enemy distance with noise
            exactEnemyDistance: enemy distance, the exact position, i.e. within 5 distances away OR know by the food enemy eaten
            exactEnemyDistanceScaredGhost: know exact enemy distance, but our ghost is scared => can be eaten => run away!
            fuzzyEnemyDistanceScaredGhost: know enemy distanse with noise, but our ghost is scared
            foodEaten: prioritize enemy that has eaten more food, if more than one enemy attacking
            distance2FoodCapsule: penalize increased distance to foodcapsule, i.e. defend capsules so don't get scared
            reverse: penalize going backwards 
        """
        return {'numberOfInvaders': -1000, 'onDefense': 100, 'stop':-100 , 'fuzzyEnemyDistance': -20, 'exactEnemyDistance': -50,
              'exactEnemyDistanceScaredGhost': -1000, 'fuzzyEnemyDistanceScaredGhost': -400, 'foodEaten': -20, 
              'distance2FoodCapsule': -5, 'reverse': -2}
    
    def getWeightsOffensive(self):
        """interpetation:
            foodScore: give reward for more score in food
            distance2food: penalize going further away from food
            distance2Home: give reward to going further away from home
            distance2capsule: try to get capsule to eat enemy ghosts
            minEnemyDistanceFuzzy: penalize getting closer to enemy, with noise
            minEnemyDistanceExact: enemy within 5 distances => run away!
            minEnemyDistanceFuzzyScared: enemy is scared, but distance is noisy, want to eat enemy, reward for getting closer, but time dependent!!!
            minEnemyDistanceExactScared: enemy scared, within 5 distances => want to eat!
            stop: penalize stopping, cause risky if enemy gets closer!
            secondEnemyDist: punish if second enemy is also close by
            reverse: penalize going backwards
            distance2EnemyWhenGhost: if ghost, then try get closer to enemy
        """
        return {'foodScore': 100, 'distance2food': -10, 'distance2Home': 1000, 'distance2capsule': -2, 'minEnemyDistanceFuzzy': -50,
                'minEnemyDistanceExact': -100, 'minEnemyDistanceFuzzyScared': 5, 'minEnemyDistanceExactScared': -20, 'stop': -100, 
                'secondEnemyDist': -10, 'reverse':-2, 'distance2EnemyWhenGhost': -5}
    
    def getWeightsReturnHome(self):
        """interpetation: 
            distance2Home: penalize higher distance to home
            distance2capsule: penalize further distance to capsule, if capsule is nearby
            minEnemyDistanceFuzzy: penalize getting closer to enemy, with noise
            minEnemyDistanceExact: high penalty for getitng within 5 distances to enemy
            minEnemyDistanceFuzzyScared: should eat enemy, but position is noisy
            minEnemyDistanceExactScared: know enemy exact position => eat them! obs: time dependent, if scared time about to run out then run away
            stop: high penalty for not moving
            onRoute: reward for going in shortest path home
            reverse: penalize going backwards
        """

        return {'distance2Home': -50,  'distance2capsule': -5, 'minEnemyDistanceFuzzy': -50, 'minEnemyDistanceExact': -100,
                'minEnemyDistanceFuzzyScared': 50, 'minEnemyDistanceExactScared': 200, 'stop': -200, 'onRoute': 30, 'reverse': -2}
        
    




    def ReturnWeights(self, gameState):
      """returns weights for the given role"""
      
      # get food i am carrying
      foodCarrying = gameState.getAgentState(self.index).numCarrying

      if self.role == "defensive":
        return self.getWeightsDefensive()
        
      # elif offensive role and food is below threshold
      elif (self.role == "offensive") and (foodCarrying < self.foodThreshold):
        return self.getWeightsOffensive()
      else: # return home
        return self.getWeightsReturnHome()
       





def GetFoodCount(foodMatrix: game.Grid)-> int:
    """given a bool matrix, counts number of 'True' elements"""
    count = 0
    for row in foodMatrix :
        for element in row:
            if element:
                count += 1
    return count
































class Terminator5000(OffensiveReflexAgent):
  def registerInitialState(self, gameState):
    
    CaptureAgent.registerInitialState(self, gameState)

    

    # bool: if on read team then true, if false then player is on blue team
    self.teamRed = gameState.isOnRedTeam(self.index) 
    #self.walls = gameState.data.layout.walls
    self.walls = gameState.getWalls()

    self.defendFood = CaptureAgent.getFoodYouAreDefending(self, gameState)
    # if self.teamRed: # i am red


  def distance2EnemyGhost(self, gameState, action):
    features = util.Counter()

    enemyPos = self.FindEnemyPosFromFood()

    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    if myState.isPacman: features['onDefense'] = 0

    # if len(enemyPos) <= 0: # don't know position for certain

      


  def FindEnemyPosFromFood(self, gameState)->List[Tuple]:
    """ if food count decreases since last call => enemy is there """

    if self.teamRed:
      myFood = gameState.getRedFood()
    else: 
      myFood = gameState.getBlueFood()
    
    enemyPos = None
    if myFood.count() != self.defendFood.count(): # if our food has decreased:
      # go through all food, find which position is missing food
      enemyPos = [(i,k) for i in range(self.defendFood.width) for k in range(self.defendFood.height) 
                  if self.defendFood[i][k] != myFood[i][k]]
    self.defendFood = myFood
    
    return enemyPos





















class NodeTree:
  def __init__(self):
    self._heuristicValue = np.inf
    self._children = [] # list of children

  def IsTerminal(self):
    """
    Check if children list is empty,
    If no children then is a terminal node
    """
    return ( len(self._children) ==0 )
  
  def ReturnHeuristicValue(self):
    """
    returns heuristic value of node
    """
    return self._heuristicValue
  
  def ReturnChildren(self):
    """
    returns list of the children of node
    """
    return self._children


def AlphaBeta(node:NodeTree, depth:int, alpha, beta, maximizingPlayer: bool):
  """ alpha beta pruning of tree """

  if depth == 0 or node.IsTerminal():
    return node.ReturnHeuristicValue()
  
  else:

    if maximizingPlayer:
      value = np.inf

      for childNode in node.ReturnChildren():
        value = max( value, AlphaBeta(childNode, depth-1,alpha,beta,False) )

        if value >= beta:
          break
        
        else:
          alpha = max(alpha, value)

    else:
      value = np.inf
      for childNode in node.ReturnChildren():
        value = min( value, AlphaBeta(childNode, depth-1,alpha,beta,False) )
        
        if value <= alpha: 
          break
        
        else:
          beta = min(beta, value)
      return value



























