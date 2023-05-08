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
from game import Directions
import game
import math
import time

import distanceCalculator
import random, time, util, sys
from util import nearestPoint
from typing import List, Tuple
import numpy as np
from baselineTeam import OffensiveReflexAgent


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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

    self.start = gameState.getAgentPosition(self.index)
    startTime_registerInitialState = time.time()
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
    
    print("number of foodcapsules are "+ str(self.originalNumberOfFood))
    # initial roles setup in start, they start at different positions, set first one to defense,second to offens
    
    self.canEatCount = self.GetEatCount(gameState)
    print(self.canEatCount)
    self.foodThreshold = int( (self.canEatCount-2)/5 )
    # print initial position of agents
    print("[id]"+str(self.index)+"initial position of agents"+ str(gameState.getAgentPosition(self.index)))
    
    # assign roles, no particular reason for this order, they are essentially in same position
    self.AssignRoles(gameState)
    print("[time] registerInitialState: "+ str(time.time()-startTime_registerInitialState))









  # function that assigns roles to agents
  def AssignRoles(self, gameState):
      """assigns roles to agents"""
      if self.index == self.getTeam(gameState)[0]:
          self.role = "defense"
      else:
          self.role = "offense"
      print("agent "+ str(self.index) + " role is " + self.role)
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
        print("closest food is "+ str(closestFood))
        
        print("closest capsule is "+ str(closestCapsule))

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
                  print("error in FindClosestFood")
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
            print("food count before enemy attack is "+ str(self.foodCountBeforeEnemyAttack) + " current food count is "+ str(currentFoodCount))
            self.foodCountBeforeEnemyAttack = currentFoodCount
            # find where food is missing

            print("self.defendfood dimensions are "+ str(self.defendFood.height) + " " + str(self.defendFood.width))

            for row in range(self.defendFood.height):
                for col in range(self.defendFood.width):
                    print("row "+ str(row) + " col "+ str(col))
                    print("self defendfood "+ str(self.defendFood[row][col]))
                    print(str(gameState.hasFood(row, col)))
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
        return {'numberOfInvaders': -1000, 'onDefense': 100, 'stop':-200 , 'fuzzyEnemyDistance': -20, 'exactEnemyDistance': -50,
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
            stop: penalize not moving
        """
        return {'foodScore': 100, 'distance2food': -10, 'distance2Home': 1000, 'distance2capsule': -2, 'minEnemyDistanceFuzzy': -50,
                'minEnemyDistanceExact': -100, 'minEnemyDistanceFuzzyScared': 5, 'minEnemyDistanceExactScared': -20, 'stop': -100, 
                'secondEnemyDist': -10, 'reverse':-2, 'distance2EnemyWhenGhost': -5, 'stop': -200}



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
            stop: penalize not moving
        """

        return {'distance2Home': -50,  'distance2capsule': -5, 'minEnemyDistanceFuzzy': -50, 'minEnemyDistanceExact': -100,
                'minEnemyDistanceFuzzyScared': 50, 'minEnemyDistanceExactScared': 200, 'stop': -200, 'onRoute': 30, 'reverse': -2, 'stop': -200}


 



  
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
              print("scared ghost")
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

  
    

  # def getWeights(self, gameState, action):
  #   return {'successorScore': 100, 'distanceToFood': -1}

  # evaluation function
  def evaluationFunction(self, gameState, mode = "Attack"):
    # heuristic function to evaluate the state of the game

    # if terminal state    
    if gameState.isOver():
      return gameState.getScore()
    
    # new heuristic function

    val = 0
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    # print("actions = ", actions)

    


    for action in actions:
      # Computes distance to invaders we can see
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

      
      #val += features * weights
      val = max(val, features * weights)
    return val/len(actions)


    # list of our food, enemy food, enemy pacman, enemy ghost, our pacman, our ghost, capsule
    foodList = self.getFood(gameState).asList()
    enemyFoodList = self.getFoodYouAreDefending(gameState).asList()
    
    enemyPacmanList = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentPosition(i)!=None]

    enemyGhostList = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if not gameState.getAgentState(i).isPacman and gameState.getAgentPosition(i)!=None]
    
    ourPacmanList = [gameState.getAgentPosition(i) for i in self.getTeam(gameState) if gameState.getAgentState(i).isPacman]
    ourPacmanList = [pacman for pacman in ourPacmanList if pacman!=None]

    ourGhostList = [gameState.getAgentPosition(i) for i in self.getTeam(gameState) if not gameState.getAgentState(i).isPacman]
    ourGhostList = [ghost for ghost in ourGhostList if ghost!=None]
    
    capsuleList = self.getCapsules(gameState)

    # find number of moves remaining in the game (max 400)
    movesRemaining = gameState.data.timeleft // 4
    # print(movesRemaining)

    # if terminal state
    if gameState.isOver():
      return gameState.getScore()
    
    # check if agent is pacman
    is_pac = gameState.getAgentState(self.index).isPacman
    # is_pac == True if agent is pacman, False if agent is ghost

    # check amount of food being carried by our team
    team_food_carried = sum([gameState.getAgentState(i).numCarrying for i in self.getTeam(gameState)])
    # print("team_food_carried = ", team_food_carried)
    # check amount of food being carried by enemy team
    enemy_food_carried = sum([gameState.getAgentState(i).numCarrying for i in self.getOpponents(gameState)])
    # print("enemy_food_carried = ", enemy_food_carried)

    # get distance to grid center
    gridCenter = (gameState.data.layout.width // 2, gameState.data.layout.height // 2)

    dist_to_grid_center = self.getMazeDistance(gameState.getAgentPosition(self.index), gridCenter)

    if dist_to_grid_center > gameState.data.layout.width // 3:
      push_to_center = 100 / dist_to_grid_center # closer to center = higher score
      
    else:
      push_to_center = 0

    pac_ghost_score = 0

    # if agent is pacman
    if is_pac:
      # prioritize eating food
      if len(foodList) > 2:
        min_food_dist = min([self.getMazeDistance(gameState.getAgentPosition(self.index), food) for food in foodList])
        pac_ghost_score += 1000 / min_food_dist

      # prioritize returning food to our side if ghost is nearby or we are carrying a lot of food
      if len(enemyGhostList) > 0 or team_food_carried > 10:
        # push to starting position
        pac_ghost_score += 1000 / self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getInitialAgentPosition(self.index))


      # prioritize eating capsules
      if len(capsuleList) > 0:
        min_capsule_dist = min([self.getMazeDistance(gameState.getAgentPosition(self.index), capsule) for capsule in capsuleList])
        pac_ghost_score += 1 / min_capsule_dist

      # prioritize eating enemy ghosts
      if len(enemyGhostList) > 0:
        # check if enemy ghost is scared
        if gameState.getAgentState(self.getOpponents(gameState)[0]).scaredTimer > 0:
          min_ghost_dist = min([self.getMazeDistance(gameState.getAgentPosition(self.index), ghost) for ghost in enemyGhostList])
          pac_ghost_score += 100 / min_ghost_dist


    # if agent is ghost
    else:
      
      # if we carry no food, push to opposite side
      if False and team_food_carried == 0:
        pac_ghost_score += 10 / (self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getInitialAgentPosition(self.index)) + 1)


      # prioritize eating enemy pacman
      if len(enemyPacmanList) > 0:
        min_pacman_dist = min([self.getMazeDistance(gameState.getAgentPosition(self.index), pacman) for pacman in enemyPacmanList])
        # check if we are scared
        if gameState.getAgentState(self.index).scaredTimer > 0:
          pac_ghost_score -= 100 / min_pacman_dist
        else:
          # pac_ghost_score += 100 / min_pacman_dist
          pass 

        # move towards central food if we are not scared
        if len(enemyFoodList) > 0:
          min_food_dist = min([self.getMazeDistance(gridCenter, food) for food in enemyFoodList])
          pac_ghost_score += 5 / min_food_dist
      




    

    return gameState.getScore() + 0.5*(team_food_carried - enemy_food_carried) + (push_to_center) + pac_ghost_score 

    

  # setup for minimax
  def getSuccessors(self, gameState, player_ID):
    """
    Returns successor states, the actions they require, and a cost of 1.
    The following successor states are the same board, but the agent
    has been moved one step in the specified direction.

    """

    # print("self.index = ", self.index)
    # print("player_ID = ", player_ID)

    successors = []
    for action in gameState.getLegalActions(agentIndex = player_ID):
      # Ignore illegal actions
      if False and (action == Directions.STOP): # avoid staying put
        continue

      successor = gameState.generateSuccessor(player_ID, action)
      successors.append((successor, action))
    return successors
  
  def max_agent(self, gameState, agent_ID, depth, time_left = math.inf, alpha = -math.inf, beta = math.inf):
    
    if depth == 0:
      return self.evaluationFunction(gameState), None
    
    if time_left < 0.05:
      # scream not enough time
      return -math.inf, None
    
    v = -math.inf
    best_action = None

    start_time = time.time()

    successor_list = self.getSuccessors(gameState, agent_ID)
    # move ordering based on heuristic
    successor_list.sort(key = lambda x: self.evaluationFunction(x[0]), reverse = True)

    blue_team = gameState.getBlueTeamIndices()
    red_team = gameState.getRedTeamIndices()

    # find agent team based on agent_ID
    if agent_ID in blue_team:
      team = blue_team
      opponent_team = red_team
    else:
      team = red_team
      opponent_team = blue_team


    for successor, action in successor_list:
      
      # check if time is up
      current_time = time.time()
      if current_time - start_time > time_left:
        return v, best_action
      else:
        time_left = time_left - (current_time - start_time)

      # check if enemy is visible from the successor state
      enemy_indices = opponent_team
      enemyList = [successor.getAgentPosition(i) for i in opponent_team]
      # remove None from enemyList, and corresponding enemy indices
      enemy_indices = [enemy_indices[i] for i in range(len(enemyList)) if enemyList[i] != None]
      enemyList = [enemy for enemy in enemyList if enemy != None]
      
      enemy_conf = None

      while (any(enemyList) and enemy_conf == None):
        # pick the closest enemy
        enemy_pos = min(enemyList, key = lambda x: self.getMazeDistance(successor.getAgentPosition(agent_ID), x))
        
        # find ID of closest enemy
        enemy_list_pos = enemyList.index(enemy_pos)
        enemy_ID = enemy_indices[enemy_list_pos]
        # find conf of enemy
        enemy_conf = successor.getAgentState(enemy_ID).configuration

        # remove enemy from list
        enemyList.remove(enemy_pos)
        enemy_indices.remove(enemy_ID)


      if enemy_conf != None:
        # print("ENEMY IS MOVING")
        # log distance to enemy
        dist = self.getMazeDistance(successor.getAgentPosition(agent_ID), enemy_conf.getPosition())
        # if dist > 5:
        act_value,_ = self.min_agent(successor, enemy_ID, depth-1, time_left, alpha, beta)

        if act_value > v:
          v = act_value
          best_action = action
        
      else:
        # if enemy is not visible, then assume board remains the same and make next move
        act_value,_ = self.max_agent(successor, agent_ID, depth-1, time_left, alpha, beta)

        if act_value > v:
          v = act_value
          best_action = action

      if v >= beta:
        return v, best_action
      alpha = max(alpha, v)
    return v, best_action
  
  def min_agent(self, gameState, agent_ID, depth, time_left = math.inf, alpha = -math.inf, beta = math.inf):
    if depth == 0:
      return self.evaluationFunction(gameState), None
    
    if time_left < 0.05:
      # scream not enough time
      return math.inf, None
    
    v = math.inf
    best_action = None
    
    start_time = time.time()


    successor_list = self.getSuccessors(gameState, agent_ID)
    # move ordering based on heuristic
    successor_list.sort(key = lambda x: self.evaluationFunction(x[0]), reverse = False) # reverse = False for min agent because we want to minimize the heuristic value - explore the least heuristic value first
    
    blue_team = gameState.getBlueTeamIndices()
    red_team = gameState.getRedTeamIndices()

    # find agent team based on agent_ID
    if agent_ID in blue_team:
      team = blue_team
      opponent_team = red_team
    else:
      team = red_team
      opponent_team = blue_team

    for successor, action in successor_list:
        
        current_time = time.time()
        if current_time - start_time > time_left - 0.05:
          return v
        else:
          time_left = time_left - (current_time - start_time)
        
        # check if enemy is visible from the successor state
        enemy_indices = opponent_team
        enemyList = [successor.getAgentPosition(i) for i in opponent_team]
        # remove None from enemyList, and corresponding enemy indices
        enemy_indices = [enemy_indices[i] for i in range(len(enemyList)) if enemyList[i] != None]
        enemyList = [enemy for enemy in enemyList if enemy != None]

        enemy_conf = None

        while (any(enemyList) and enemy_conf == None):
          # pick the closest enemy
          enemy_pos = min(enemyList, key = lambda x: self.getMazeDistance(successor.getAgentPosition(self.index), x))
          # if enemy is visible, then assume enemy makes next move

          # find ID of closest enemy
          enemy_list_pos = enemyList.index(enemy_pos)
          enemy_ID = enemy_indices[enemy_list_pos]

          # find conf of enemy
          enemy_conf = successor.getAgentState(enemy_ID).configuration

          # remove enemy from list
          enemyList.remove(enemy_pos)
          enemy_indices.remove(enemy_ID)

          
        if enemy_conf != None:
          act_value,_ = self.max_agent(successor, enemy_ID, depth-1, time_left, alpha, beta)

          if act_value < v:
            v = act_value
            best_action = action  
        
        else:
          # if enemy is not visible, then assume board remains the same and make next move
          act_value,_ = self.min_agent(successor, agent_ID, depth-1, time_left, alpha, beta)

          if act_value < v:
            v = act_value
            best_action = action
  
        if v <= alpha:
          return v, best_action
        beta = min(beta, v)
    return v, best_action
  
  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    # depth_list = [4, 6, 8]
    depth_list = [3] # no IDS for now

    time_left = 0.95 # time left for the agent to choose an action

    depth_act_dict = {}

    # iteratively deepening search
    start_time = time.time()

    for depth in depth_list:
      current_time = time.time()
      if current_time - start_time > time_left:
        break
      
      v, best_action = self.max_agent(gameState, self.index, depth, time_left)
      if best_action != None:
        depth_act_dict[depth] = best_action
      else:
        break

    # print("depth_act_dict = ", depth_act_dict)
    if depth_act_dict == {}:
      return random.choice(actions)
    else:
      # choose action with highest depth
      best_depth = max(depth_act_dict.keys())
      best_action = depth_act_dict[best_depth]
      return best_action    
    


def GetFoodCount(foodMatrix: game.Grid)-> int:
    """given a bool matrix, counts number of 'True' elements"""
    count = 0
    for row in foodMatrix :
        for element in row:
            if element:
                count += 1
    return count