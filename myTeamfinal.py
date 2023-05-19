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
from typing import List, Tuple # for type hinting


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAttackAgent', second = 'DummyDefenseAgent'):
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

class DummyAttackAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    startInit = time.time()
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

    # =====original register initial state=======
    self.start = gameState.getAgentPosition(self.index)
    self.cross_dist = 0

    # find symmetric start position of opponent
    self.opponent_start = (gameState.data.layout.width - 1 - self.start[0], gameState.data.layout.height - 1 - self.start[1])
    self.isFull = False

    # reset dictionaries
    self.min_agent_dict = {}
    self.max_agent_dict = {}
    self.heurvalue_dict = {}

    # find points on center line
    self.center_line = []
    for i in range(1, gameState.data.layout.height-1):
        if not gameState.hasWall(gameState.data.layout.width//2 - 1, i) and not gameState.hasWall(gameState.data.layout.width//2, i) and not gameState.hasWall(gameState.data.layout.width//2 + 1, i):
            self.center_line.append((gameState.data.layout.width/2, i))


    self.center_dist_from_pos_dict = {}
    # for all traversable positions on the map
    for i in range(gameState.data.layout.width):
      for j in range(gameState.data.layout.height):
        if not gameState.hasWall(i, j):
          # calculate distance to center line
          wow_dist = min([self.getMazeDistance((i, j), k) for k in self.center_line])
          self.center_dist_from_pos_dict[(i, j)] = (wow_dist, 1 / (wow_dist + 0.01) )

    # find total time steps
    self.total_time = gameState.data.timeleft

    # find dead ends
    self.FindDeadEnds(gameState)
    # print("Time to initialize: ", time.time() - startInit)

    # get number of keys in dead end dictionary
    self.dead_end_keys = len(self.deadEndQuantizationDict.keys())
    # print("Number of dead ends: ", self.dead_end_keys)


    # debug confirmation of all points in dictionary
    # startPos = (gameState.data.layout.width//2)
    # endPos = gameState.data.layout.width - 1
    # for x in range(startPos, endPos):
    #   for y in range(1,gameState.data.layout.height - 1):
    #     #if not wall
    #     if not gameState.hasWall(x,y):
    #       # check if self.pointsInDeadEndPaths.keys()
    #       positionInfo = self.pointsInDeadEndPaths[(x,y)]
    #       if positionInfo.point ==None:
    #         # draw yellow
    #         self.debugDraw([(x,y)], [1,0,0], clear=False)
    #       else:
    #         # draw green
    #         self.debugDraw([(x,y)], [0,1,0], clear=False)
          
            
    






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

  # evaluation function
  def evaluationFunction(self, gameState):
    # heuristic function to evaluate the state of the game

    # if terminal state    
    if gameState.isOver():
      val = gameState.getScore()
      return val
    
    # initialize variables for food carrying
    TeamFoodCarrying = 0
    EnemyFoodCarrying = 0
    
    # bool, for when pacman should deposit food
    self.isFull = gameState.getAgentState(self.index).numCarrying >= 5 # todo

    # get food carrying of team
    for i in self.getTeam(gameState):
      TeamFoodCarrying += gameState.getAgentState(i).numCarrying
    
    # get food carrying of enemy
    for i in self.getOpponents(gameState):
      EnemyFoodCarrying += gameState.getAgentState(i).numCarrying
    
    # score heuristic
    val = gameState.getScore() * 1000 # score is important

    # ------------food carrying heuristic---------

    # define explore vs exploit factor
    time_left = gameState.data.timeleft
    # total time left is self.total_time

    # start at 1000, reduce to 250 around half time
    explore_v_exploit = 500


    food_carry_val = (TeamFoodCarrying - EnemyFoodCarrying) * explore_v_exploit # food carrying is important

    # decay factor for food carrying: want to make depositing food more important as amount of carried food increases

    val += food_carry_val
    #------------------------------------------------

    # check if i am pacman
    amPac = gameState.getAgentState(self.index).isPacman

    # distance between starting position and current position
    start_dist = self.getMazeDistance(self.start, gameState.getAgentPosition(self.index))

    # distance to center line
    center_dist = self.center_dist_from_pos_dict[gameState.getAgentPosition(self.index)][0]

    start_dist = - center_dist


    # i'm a ghost, but I am attack agent => push to other side
    if not amPac:
      # push to the other side
      val -= center_dist

      # chase enemy pacman and eat it
      enemyList = self.getOpponents(gameState)
      # check list of enemy pacmans
      enemyPacList = [gameState.getAgentState(i) for i in enemyList if (True or gameState.getAgentState(i).isPacman) and gameState.getAgentState(i).getPosition() != None]
      
      # incentivize eating enemy pacman
      val += 100 /(len(enemyPacList) + 1)

      enemyList = [gameState.getAgentPosition(i) for i in enemyList]
      # remove None values
      enemy_pos_list = [i for i in enemyList if i != None]

      if len(enemy_pos_list) > 0:
        enemy_dist_list = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in enemy_pos_list]

        for i in range(len(enemy_dist_list)):
            val -= enemy_dist_list[i] * 250
      # return value
      return val


    else: # if i am pacman, i.e., i am on the other side
      
      val += center_dist + 200

      # check how much food i have in my stomach
      foodCarrying = gameState.getAgentState(self.index).numCarrying
      # if enough food in my stomach, go back to my side
      
      # define imperative in terms of food carrying
      imperative_to_return = max(0, (foodCarrying - 3) / 5 )

      val -= center_dist * imperative_to_return

      # find food and eat it
      foodList = self.getFood(gameState).asList()

      #check if there are any capsules to eat
      capsuleList = self.getCapsules(gameState)

      # if there are capsules
      val += 1000 / (len(capsuleList) + 1)

      if len(capsuleList) > 0:
          capsule_dist_list = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in capsuleList]

          for i in range(len(capsule_dist_list)):
            val += 100/capsule_dist_list[i]
        
        # todo: 2 rewards, capsule more valuable, compare values, values depending on distance + gain

        # find closest food
        # closest_food_dist = min(food_dist_list)
        # closest_food_dist = min(closest_food_dist, closest_capsule_dist)
        # val += 50/closest_food_dist

      # check list of enemy ghosts
      enemyList = self.getOpponents(gameState)
      enemyGhostList = [gameState.getAgentState(i) for i in enemyList if not gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]

      # find distance to enemy ghosts
      enemyGhostPosList = [i.getPosition() for i in enemyGhostList]
      
      enemyGhostDistList = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in enemyGhostPosList]

      # bool list of whether enemy ghost is scared
      enemyScaredList = []
      for i, indexEnemy in enumerate(enemyGhostDistList):
        if gameState.getAgentState(i).scaredTimer > enemyGhostDistList[i]:
          enemyScaredList.append(True)
        else:
          enemyScaredList.append(False)

      # make list of distances to scared enemy ghosts
      # scaredEnemyGhostList = [enemyGhostList[i] for i in range(len(enemyGhostList)) if enemyScaredList[i]]
      # scaredEnemyGhostPosList = [enemyGhostPosList[i] for i in range(len(enemyGhostPosList)) if enemyScaredList[i]]
      scaredEnemyGhostDistList = [enemyGhostDistList[i] for i in range(len(enemyGhostDistList)) if enemyScaredList[i]]

      # make list of distances to unscared enemy ghosts
      # unscaredEnemyGhostList = [enemyGhostList[i] for i in range(len(enemyGhostList)) if not enemyScaredList[i]]
      unscaredEnemyGhostPosList = [enemyGhostPosList[i] for i in range(len(enemyGhostPosList)) if not enemyScaredList[i]]
      unscaredEnemyGhostDistList = [enemyGhostDistList[i] for i in range(len(enemyGhostDistList)) if not enemyScaredList[i]]

      # # if no enemy ghosts are scared, then run away
      # if not any(enemyScaredList):
      #     for i in range(len(enemyGhostDistList)):
      #         val += enemyGhostDistList[i] * 100
      # else:
      #   val -=  1000 / min(scaredEnemyGhostDistList)

      chase_factor = 1/(len(enemyGhostPosList) + 1)

      if len(enemyGhostPosList) > 0:
        enemyGhostDistList = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in enemyGhostPosList]

        if min(enemyGhostDistList) <= 2:
          chase_factor = 0.1

        # run from unscared enemy ghosts
        for i in range(len(unscaredEnemyGhostDistList)):
          val += unscaredEnemyGhostDistList[i]

        # chase scared enemy ghosts
        for i in range(len(scaredEnemyGhostDistList)):
          val += 1000 / scaredEnemyGhostDistList[i]

        # is the closest unscared ghost to my left or right?
        if len(unscaredEnemyGhostDistList) > 0:
          closest_unscared_ghost_index = unscaredEnemyGhostDistList.index(min(unscaredEnemyGhostDistList))
          closest_unscared_ghost_pos = unscaredEnemyGhostPosList[closest_unscared_ghost_index]

          # check if ghost is closer to grid center than I am
          ghost_to_center_dist = self.center_dist_from_pos_dict[closest_unscared_ghost_pos][0]
        
          if ghost_to_center_dist > center_dist:
            val -= center_dist * 1.5

      if len(foodList) > 2:
        food_dist_list = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in foodList]

        # incorporate information about dead ends
        if chase_factor != 1:
          for i in range(len(foodList)):
            wow_object = self.pointsInDeadEndPaths[foodList[i]]
            if wow_object.point != None:
              food_dist_list[i] += 200 * ((1 + wow_object.indexFromStart) / wow_object.lengthOfDeadEnd) / chase_factor

        # sort food list by distance
        # food_dist_list.sort()
        # retain only 5 closest food
        # food_dist_list = food_dist_list[:3]
        
        food_val = 0
        food_decay_factor = math.exp(-foodCarrying/6)
        for i in range(len(food_dist_list)):
          food_val += (50/food_dist_list[i]) * food_decay_factor * chase_factor

        val += food_val

      else:
        val -= center_dist * 1.5



    # get possible actions from current state
    actions = gameState.getLegalActions(self.index)
    # check if possible actions are stor or reverse
    reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    # check if reverse is in possible actions and if stop is in possible actions and that possible actions has length 2
    # penalizing if got to dead ends
    if reverse in actions and len(actions) == 2 and Directions.STOP in actions:
      carryingFood = gameState.getAgentState(self.index).numCarrying
      # less penalty if not carrying a lot of food
      val -= math.exp(carryingFood) 
    


    return val

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

  def max_agent(self, gameState, agent_ID, depth, total_compute_time = math.inf, alpha = -math.inf, beta = math.inf):
    
    if depth == 0:
      return self.evaluationFunction(gameState), None
    
    # hash (gameState, agent_ID, depth) and store (value, action) associated with this in a dictionary
    hash_val = hash((gameState, agent_ID, depth))
    # check if this hash already exists in the dictionary
    if hash_val in self.max_agent_dict:
      return self.max_agent_dict[hash_val]


    if total_compute_time < 0.001:
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
      if current_time - start_time > total_compute_time:
        return -math.inf, None
        # time_left = time_left - (current_time - start_time)

      # check if enemy is visible from the successor state
      enemy_indices = opponent_team

      # print("enemy_indices = ", enemy_indices)

      enemy_pos_list = [successor.getAgentPosition(i) for i in enemy_indices]
      # print("enemy_pos_list = ", enemy_pos_list)
      # filter out none positions
      enemy_indices = [enemy_indices[i] for i in range(len(enemy_pos_list)) if enemy_pos_list[i] != None]
      enemy_pos_list = [enemy_pos for enemy_pos in enemy_pos_list if enemy_pos != None]

      # distance threshold for enemy visibility: 5 units
      # dist_list = [self.getMazeDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]

      # manhattan distance instead of maze distance
      dist_list = [util.manhattanDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]

      # filter out enemies that are too far away
      enemy_indices = [enemy_indices[i] for i in range(len(dist_list)) if dist_list[i] < 6]
      enemyList = [successor.getAgentPosition(i) for i in enemy_indices]

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
        current_time = time.time()
        time_left = total_compute_time - (current_time - start_time)
        act_value,_ = self.min_agent(successor, enemy_ID, depth-1, time_left, alpha, beta)

        # check if min agent ran out of time
        if act_value == math.inf:
          # return v, best_action
          return -math.inf, None

        if act_value > v:
          v = act_value
          best_action = action
        
      else:
        # if enemy is not visible, then assume board remains the same and make next move
        current_time = time.time()
        time_left = total_compute_time - (current_time - start_time)
        act_value,_ = self.max_agent(successor, agent_ID, depth-1, time_left, alpha, beta)

        # check if max agent ran out of time
        if act_value == -math.inf:
          # return v, best_action
          return -math.inf, None


        if act_value > v:
          v = act_value
          best_action = action

      if v >= beta:
        # add to dictionary
        self.max_agent_dict[hash_val] = (v, best_action)
        return v, best_action
      alpha = max(alpha, v)
    
    # add to dictionary if the value is not -inf
    if v != -math.inf:
      self.max_agent_dict[hash_val] = (v, best_action)
    return v, best_action
  
  def min_agent(self, gameState, agent_ID, depth, total_compute_time = math.inf, alpha = -math.inf, beta = math.inf):
    if depth == 0:
      return self.evaluationFunction(gameState), None

    hash_val = hash((gameState, agent_ID, depth))
    # check if this hash already exists in the dictionary
    if hash_val in self.min_agent_dict:
      return self.min_agent_dict[hash_val]

    if total_compute_time < 0.0001:
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
        if current_time - start_time > total_compute_time:
          return math.inf, None
          # total_compute_time = total_compute_time - (current_time - start_time)
        
        # check if enemy is visible from the successor state
        enemy_indices = opponent_team

        # print("enemy_indices = ", enemy_indices)

        enemy_pos_list = [successor.getAgentPosition(i) for i in enemy_indices]
        # print("enemy_pos_list = ", enemy_pos_list)
        # filter out none positions
        enemy_indices = [enemy_indices[i] for i in range(len(enemy_pos_list)) if enemy_pos_list[i] != None]
        enemy_pos_list = [enemy_pos for enemy_pos in enemy_pos_list if enemy_pos != None]

        # distance threshold for enemy visibility: 5 units
        # dist_list = [self.getMazeDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]
        # manhattan distance instead of maze distance
        dist_list = [util.manhattanDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]

        # filter out enemies that are too far away
        enemy_indices = [enemy_indices[i] for i in range(len(dist_list)) if dist_list[i] < 6]
        enemyList = [successor.getAgentPosition(i) for i in enemy_indices]

        
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
          current_time = time.time()
          time_left = total_compute_time - (current_time - start_time)
          act_value,_ = self.max_agent(successor, enemy_ID, depth-1, time_left, alpha, beta)

          # check if max agent ran out of time
          if act_value == -math.inf:
            # return v, best_action
            return math.inf, None


          if act_value < v:
            v = act_value
            best_action = action  
        
        else:
          # if enemy is not visible, then assume board remains the same and make next move
          current_time = time.time()
          time_left = total_compute_time - (current_time - start_time)
          act_value,_ = self.min_agent(successor, agent_ID, depth-1, time_left, alpha, beta)

          # check if min agent ran out of time
          if act_value == math.inf:
            # return v, best_action
            return math.inf, None


          if act_value < v:
            v = act_value
            best_action = action
  
        if v <= alpha:
          # add to dictionary
          self.min_agent_dict[hash_val] = (v, best_action)
          return v, best_action
        beta = min(beta, v)
    
    # add to dictionary if the value is not inf
    if v != math.inf:
      self.min_agent_dict[hash_val] = (v, best_action)
    return v, best_action

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    depth_list = [2, 6, 10, 14] # best IDS Scenario
    # depth_list = [10] # no IDS for now 

    total_compute_time = 0.995 # time left for the agent to choose an action

    depth_act_dict = {}

    # reset dictionaries
    self.max_agent_dict = {}
    self.min_agent_dict = {}
    self.heurvalue_dict = {}


    # iteratively deepening search
    start_time = time.time()

    for depth in depth_list:
      # time bookkeeping      
      current_time = time.time()
      time_spent = current_time - start_time
      time_left = total_compute_time - time_spent

      if time_left < 0.001:
        break
      
      v, best_action = self.max_agent(gameState, self.index, depth, time_left)
      if best_action != None:
        depth_act_dict[depth] = best_action
      else:
        break

    # print("depth_act_dict = ", depth_act_dict)
    if depth_act_dict == {}:
      # print("Time taken by Offense = ", time.time() - start_time)
      print("Offense is Random")
      return random.choice(actions)
    else:
      # choose action with highest depth
      best_depth = max(depth_act_dict.keys())
      best_action = depth_act_dict[best_depth]

      # print time taken for agent to choose action
      # print("Time taken by Offense = ", time.time() - start_time)
      # print("best_depth = ", best_depth)

      return best_action    

  def FindDeadEnds(self, gameState)->list:
    """Finds dead ends in the maze. Returns a list of dead ends."""
    
    walls = gameState.getWalls()
    dead_ends = []
    leadingToDeadEnd = []

    # dictionary of DeadEndQuantization objects
    self.deadEndQuantizationDict = {}

    # dictionary of points in dead end paths
    self.pointsInDeadEndPaths = {}

    # list to save points not regarded as dead ends
    self.pointsNotDeadEnds = []
    
    # check if agent is red or blue so know x ranges
    if self.red:
      startPos = (walls.width)//2
      endPos = walls.width - 1
    else:
      startPos = 1
      endPos = (walls.width)//2
      

    # iterate through all cells in the maze of opponent side
    for x in range(startPos, endPos):
      for y in range(1, walls.height - 1):
        if not walls[x][y]:

          # if the number of adjacent cells that are walls is greater than 2, then it is a dead end
          if len([i for i in self.GetAdjacentCells((x, y), walls)   ]) == 1: # if not walls[i[0]][i[1]]
            dead_ends.append((x, y))

            # find all cells that lead to this dead end
            leadingToDeadEnd.append(self.GetAdjacentCells((x, y), walls))

            listLeading2DeadEnds = self.FindPointsFromDeadEnd((x, y), walls)

            # add last point in listLeading2DeadEnds to dictionary of DeadEndQuantization objects
            self.deadEndQuantizationDict[listLeading2DeadEnds[-1]] = DeadEndQuantization(listLeading2DeadEnds[-1], len(listLeading2DeadEnds))

            # entry point to dead end path set to none, add to pointsInDeadEndPaths but as None
            self.pointsInDeadEndPaths[listLeading2DeadEnds[-1]] = DeadEndQuantization(None, None)
            self.pointsInDeadEndPaths[(x,y)] = DeadEndQuantization(None, None)

            # color point yellow, drawing point that is first point in dead end path
            # self.debugDraw(listLeading2DeadEnds[-1], [1,1,0], clear=False)

            # draw red dots on dead ends
            # self.debugDraw(dead_ends[-1], [1,0,0], clear=False)
            

            # loop through all points in listLeading2DeadEnds backwards and add to dictionary   [new]
            for i in range(len(listLeading2DeadEnds)-1, -1, -1):
              # temporary deadendquantization object
              tempDEQ = DeadEndQuantization(listLeading2DeadEnds[i], len(listLeading2DeadEnds))
              # print(str(listLeading2DeadEnds[i]))
              
              # since going backwards, the index of the point in the list is len(listLeading2DeadEnds)-1-i, indexes left to dead end is i, so point i is last point before dead end
              tempDEQ.Leading2DeadEndInformation(  len(listLeading2DeadEnds)-i, i+1) 


              # print("index i = ", str(len(listLeading2DeadEnds)-i)+ " until dead end is "+ str(i+1)  )
              self.pointsInDeadEndPaths[listLeading2DeadEnds[i]] = tempDEQ

              # color point green
              # self.debugDraw(listLeading2DeadEnds[i], [0,1,0], clear=False)
            
          # else:
          else: # not dead ends, to be added to pointsInDeadEndPaths but as None
            self.pointsNotDeadEnds.append((x, y))
        # else: # is wall
        #     self.pointsNotDeadEnds.append((x, y))



            # self.deadEndQuantizationDict[(x, y)] = DeadEndQuantization(None, None)
          #   # point is not a dead end, we add it to dictionary but set position as None
          #   if not (x, y) in self.deadEndQuantizationDict:
          #     self.deadEndQuantizationDict[(x, y)] = DeadEndQuantization(None, None)
          #     # color magenta
          #     self.debugDraw((x, y), [1,0,1], clear=False)

      # loop through pointsNotDeadEnds and check if they are in deadEndQuantizationDict
      # if not, add to deadEndQuantizationDict
    
    for point in self.pointsNotDeadEnds:
      # print(str(point))
      if not point in self.deadEndQuantizationDict and not point in self.pointsInDeadEndPaths:
        # add points that are not in dead end path to dictionary but set position as None
        self.pointsInDeadEndPaths[point] = DeadEndQuantization(None, None)
        
        # color magenta
        # self.debugDraw(point, [1,0,1], clear=False)
    

    return dead_ends

  def GetAdjacentCells(self, cell:Tuple, walls):
    """Returns a list of adjacent cells that are not walls."""
    x, y = cell
    adjacentCells = []

    # if both x and y are not on the edge of the board
    # if x <walls.width - 1 and y < walls.height - 1 and x > 0 and y > 0:
    if not walls[x+1][y]:
      adjacentCells.append((x+1, y))
    if not walls[x-1][y]:
      adjacentCells.append((x-1, y))
    if not walls[x][y+1]:
      adjacentCells.append((x, y+1))
    if not walls[x][y-1]:
      adjacentCells.append((x, y-1))
    return adjacentCells
  
  def FindPointsFromDeadEnd(self, cell:Tuple, walls):
    """starts at a dead end, saves points in list until number of adjasent cells is greater than 2"""
    x, y = cell

    # draw red dots on dead ends
    # self.debugDraw((x,y), [1,0,0], clear=False)


    points = []
    # points.append((x, y))

    # dictionary of already visited cells
    visitedCells = {}

    # add current cell to visited cells
    visitedCells[(x, y)] = True

    while len(self.GetAdjacentCells((x, y), walls)) < 3 :
      listOfAdjacentCells = self.GetAdjacentCells((x, y), walls)
      
      # x, y = self.GetAdjacentCells((x, y), walls)[0]

      # find the unvisited adjacent cell
      for i in listOfAdjacentCells:
        if i not in visitedCells:
          x, y = i

          visitedCells[(x, y)] = True
          break

      points.append((x, y))

      # draw orange dots on points on points leading to dead ends
      # self.debugDraw(points[-1], [1,0.5,0], clear=False)
      

    # last point colored blue: entry point, point that has multiple actions but is adjasent to a dead end path
    # self.debugDraw(points[-1], [0,0,1], clear=False)

    # pop last point
    if len(points) > 1:
      points.pop()
    

    return points


class DeadEndQuantization:
  def __init__(self, point:Tuple, lengthOfDeadEnd:int):
    self.point = point
    self.lengthOfDeadEnd = lengthOfDeadEnd

  def Leading2DeadEndInformation(self, indexFromStart:int, index2End:int): #[new]
    """Saves information about index position relative to start and end of dead end path."""
    self.indexFromStart = indexFromStart
    self.index2End = index2End
    



class DummyDefenseAgent(CaptureAgent):
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

    # =====original register initial state=======
    self.start = gameState.getAgentPosition(self.index)
    self.cross_dist = 0

    # find symmetric start position of opponent
    self.opponent_start = (gameState.data.layout.width - 1 - self.start[0], gameState.data.layout.height - 1 - self.start[1])

    # to infer position of enemy by tracking food availability
    self.remainingFoodToDefend = self.getFoodYouAreDefending(gameState).asList()
    self.remainingPowerPillsToDefend = self.getCapsulesYouAreDefending(gameState)
    self.last_seen = []    

    # reset dictionaries
    self.min_agent_dict = {}
    self.max_agent_dict = {}
    self.heurvalue_dict = {}

    # find points on center line
    self.center_line = []
    for i in range(1, gameState.data.layout.height-1):
        if not gameState.hasWall(gameState.data.layout.width//2 - 1, i) and not gameState.hasWall(gameState.data.layout.width//2, i) and not gameState.hasWall(gameState.data.layout.width//2 + 1, i):
            self.center_line.append((gameState.data.layout.width/2, i))

    self.prev_enemy_pac_list = []
    self.prev_min_pac_dist = math.inf

    self.kill_timer = 0


    self.center_dist_from_pos_dict = {}

    # for all traversable positions on the map
    for i in range(gameState.data.layout.width):
      for j in range(gameState.data.layout.height):
        if not gameState.hasWall(i, j):
          # calculate distance to center line
          wow_dist = min([self.getMazeDistance((i, j), k) for k in self.center_line])
          self.center_dist_from_pos_dict[(i, j)] = (wow_dist, 1 /(wow_dist + 0.01))

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

  # evaluation function
  def evaluationFunction(self, gameState):
    # heuristic function to evaluate the state of the game

    # if terminal state    
    if gameState.isOver():
      val = gameState.getScore()
      return val
    
    TeamFoodCarrying = 0
    EnemyFoodCarrying = 0
    # get food carrying of team
    for i in self.getTeam(gameState):
      TeamFoodCarrying += gameState.getAgentState(i).numCarrying
    # get food carrying of enemy
    for i in self.getOpponents(gameState):
      EnemyFoodCarrying += gameState.getAgentState(i).numCarrying
    
    val = gameState.getScore() * 1000 # score is important

    food_carry_val = - EnemyFoodCarrying * 500 # food carrying is important | only care about enemy food carrying

    # the more food an enemy has, the more important it is to eat it
    # food_carry_val = food_carry_val * math.exp(float(TeamFoodCarrying)/5)
    val += food_carry_val


    # check if i am pacman
    amPac = gameState.getAgentState(self.index).isPacman

    # check if I am scared
    amScared = gameState.getAgentState(self.index).scaredTimer > 0


    # use kill timer
    # if self.kill_timer > 0:
      # self.kill_timer -= 1
      # value of a kill starts at 100000 and decreases as time passes
      # val += 100000 - self.kill_timer * 10000


    if amPac:

      # need to implement some scenario where it is useful to go to the other side
      val = -1000000 # pacman should not be on defense
      return val
    
    # find food you are defending
    foodList = self.getFoodYouAreDefending(gameState).asList()

    # if foodList != self.remainingFoodToDefend:
    #   # compare self.remainingFoodToDefend and foodList to find out which food was eaten
    #   self.last_seen = [i for i in self.remainingFoodToDefend if i not in foodList]

    pillList = self.getCapsulesYouAreDefending(gameState)

    # if pillList != self.remainingPowerPillsToDefend:
    #   # compare self.remainingPowerPillsToDefend and pillList to find out which pill was eaten
    #   eaten_pills = [i for i in self.remainingPowerPillsToDefend if i not in pillList]

    #   # append to last_seen
    #   self.last_seen += eaten_pills
    
    # update self.remainingFoodToDefend
    # self.remainingFoodToDefend = foodList

    # chase enemy pacman and eat it
    enemyList = self.getOpponents(gameState)
    # check list of enemy pacmans
    enemyPacList = [i for i in enemyList if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]


    enemy_pos_list = [gameState.getAgentPosition(i) for i in enemyPacList] # already filtered out None positions in enemyPacList

    enemy_dist_list = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in enemy_pos_list]


    # code from offensive agent

    # # incentivize eating enemy pacman
    if not amScared:

      val += 1000 /(len(enemyPacList) + 1)
      enemyList = [gameState.getAgentPosition(i) for i in enemyList]
      # remove None values
      for i in range(len(enemy_dist_list)):
        val -= enemy_dist_list[i] * 250

    # end of code from offensive agent


    # incentivize eating enemy pacman
    # if not amScared:
    #   val += 1000 * len(enemyPacList)
    # else:
    #   val -= 1000 * len(enemyPacList)

    # if len(enemy_pos_list) > 0 and not amScared:
    #   for i in range(len(enemy_dist_list)):
    #     # find out how much food enemy has
    #     food_carried = gameState.getAgentState(enemyList[i]).numCarrying
    #     val += (50/ enemy_dist_list[i]) * math.exp(food_carried/3) # the more food enemy has, the more important it is to eat it

    # if (len(enemy_pos_list) == len(self.prev_enemy_pac_list) -1) and not amScared:
    #   if self.prev_min_pac_dist == 1:
    #     # this means you just ate an enemy pacman
    #     val += 10000
    #     # self.kill_timer = 10
    #   elif self.prev_min_pac_dist == 5:
    #     val -= 10000

    # # reset self.prev_enemy_pac_list
    # self.prev_enemy_pac_list = enemy_pos_list
    
    # if len(enemy_dist_list) > 0:
    #   self.prev_min_pac_dist = min(enemy_dist_list)
    # else:
    #   self.prev_min_pac_dist = math.inf

    if amScared:
      val -= 20000 # try not to be scared  
      enemy_dist_list = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in enemy_pos_list]
      for i in range(len(enemy_dist_list)):
          val += enemy_dist_list[i] * 250

    # move to minimize distance between positions in self and self.last_seen
    if len(self.last_seen) > 0 and not amScared:
      last_seen_dist = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in self.last_seen]

      value_of_position = []

      for i in range(len(last_seen_dist)):
        dist_to_food_list = [self.getMazeDistance(self.last_seen[i], food_item) for food_item in self.remainingFoodToDefend] 
        # filter out entries with distance > 10
        dist_to_food_list = [dist for dist in dist_to_food_list if dist < 10]

        value_of_position.append(len(dist_to_food_list))

      # find more valuable position idx
      idx = value_of_position.index(max(value_of_position))

      val += 5000/(last_seen_dist[idx] + 1) # the closer you are to the last seen position, the better


    if len(enemy_pos_list) == 0 and len(self.last_seen) == 0 and not amScared:
      # move to centered food
      defend_food_dist_list = [self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in self.remainingFoodToDefend]

      # find least distance to center line for each food in defend_food_dist_list

      food_risk = [self.center_dist_from_pos_dict[i][1] for i in self.remainingFoodToDefend]

      # multiply food_risk by defend_food_dist_list
      mult_list = [food_risk[i] * defend_food_dist_list[i] for i in range(len(food_risk))]
    
      val -= sum(mult_list) * 50
 
    # return value
    return val

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
  
  def max_agent(self, gameState, agent_ID, depth, total_compute_time = math.inf, alpha = -math.inf, beta = math.inf):
    
    if depth == 0:
      return self.evaluationFunction(gameState), None
    
    # hash (gameState, agent_ID, depth) and store (value, action) associated with this in a dictionary
    hash_val = hash((gameState, agent_ID, depth))
    # check if this hash already exists in the dictionary
    if hash_val in self.max_agent_dict:
      return self.max_agent_dict[hash_val]



    if total_compute_time < 0.001:
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
      if current_time - start_time > total_compute_time:
        return -math.inf, None
        # time_left = time_left - (current_time - start_time)

      # check if enemy is visible from the successor state
      enemy_indices = opponent_team

      # print("enemy_indices = ", enemy_indices)

      enemy_pos_list = [successor.getAgentPosition(i) for i in enemy_indices]
      # print("enemy_pos_list = ", enemy_pos_list)
      # filter out none positions
      enemy_indices = [enemy_indices[i] for i in range(len(enemy_pos_list)) if enemy_pos_list[i] != None]
      enemy_pos_list = [enemy_pos for enemy_pos in enemy_pos_list if enemy_pos != None]

      # distance threshold for enemy visibility: 5 units
      # dist_list = [self.getMazeDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]
      # manhattan distance instead of maze distance
      dist_list = [util.manhattanDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]

      # filter out enemies that are too far away
      enemy_indices = [enemy_indices[i] for i in range(len(dist_list)) if dist_list[i] < 6]
      enemyList = [successor.getAgentPosition(i) for i in enemy_indices]

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
        current_time = time.time()
        time_left = total_compute_time - (current_time - start_time)
        act_value,_ = self.min_agent(successor, enemy_ID, depth-1, time_left, alpha, beta)

        # check if min agent ran out of time
        if act_value == math.inf:
          # return v, best_action
          return -math.inf, None

        if act_value > v:
          v = act_value
          best_action = action
        
      else:
        # if enemy is not visible, then assume board remains the same and make next move
        current_time = time.time()
        time_left = total_compute_time - (current_time - start_time)
        act_value,_ = self.max_agent(successor, agent_ID, depth-1, time_left, alpha, beta)

        # check if max agent ran out of time
        if act_value == -math.inf:
          # return v, best_action
          return -math.inf, None


        if act_value > v:
          v = act_value
          best_action = action

      if v >= beta:
        # add to dictionary
        self.max_agent_dict[hash_val] = (v, best_action)
        return v, best_action
      alpha = max(alpha, v)
    
    # add to dictionary if the value is not -inf
    if v != -math.inf:
      self.max_agent_dict[hash_val] = (v, best_action)
    return v, best_action
  
  def min_agent(self, gameState, agent_ID, depth, total_compute_time = math.inf, alpha = -math.inf, beta = math.inf):
    if depth == 0:
      return self.evaluationFunction(gameState), None

    hash_val = hash((gameState, agent_ID, depth))
    # check if this hash already exists in the dictionary
    if hash_val in self.min_agent_dict:
      return self.min_agent_dict[hash_val]

    if total_compute_time < 0.0001:
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
        if current_time - start_time > total_compute_time:
          return math.inf, None
          # total_compute_time = total_compute_time - (current_time - start_time)
        
        # check if enemy is visible from the successor state
        enemy_indices = opponent_team

        # print("enemy_indices = ", enemy_indices)

        enemy_pos_list = [successor.getAgentPosition(i) for i in enemy_indices]
        # print("enemy_pos_list = ", enemy_pos_list)
        # filter out none positions
        enemy_indices = [enemy_indices[i] for i in range(len(enemy_pos_list)) if enemy_pos_list[i] != None]
        enemy_pos_list = [enemy_pos for enemy_pos in enemy_pos_list if enemy_pos != None]

        # distance threshold for enemy visibility: 5 units
        # dist_list = [self.getMazeDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]
        # manhattan distance instead of maze distance
        dist_list = [util.manhattanDistance(successor.getAgentPosition(agent_ID), enemy_pos) for enemy_pos in enemy_pos_list]

        # filter out enemies that are too far away
        enemy_indices = [enemy_indices[i] for i in range(len(dist_list)) if dist_list[i] < 6]
        enemyList = [successor.getAgentPosition(i) for i in enemy_indices]

        
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
          current_time = time.time()
          time_left = total_compute_time - (current_time - start_time)
          act_value,_ = self.max_agent(successor, enemy_ID, depth-1, time_left, alpha, beta)

          # check if max agent ran out of time
          if act_value == -math.inf:
            # return v, best_action
            return math.inf, None


          if act_value < v:
            v = act_value
            best_action = action  
        
        else:
          # if enemy is not visible, then assume board remains the same and make next move
          current_time = time.time()
          time_left = total_compute_time - (current_time - start_time)
          act_value,_ = self.min_agent(successor, agent_ID, depth-1, time_left, alpha, beta)

          # check if min agent ran out of time
          if act_value == math.inf:
            # return v, best_action
            return math.inf, None


          if act_value < v:
            v = act_value
            best_action = action
  
        if v <= alpha:
          # add to dictionary
          self.min_agent_dict[hash_val] = (v, best_action)
          return v, best_action
        beta = min(beta, v)
    
    # add to dictionary if the value is not inf
    if v != math.inf:
      self.min_agent_dict[hash_val] = (v, best_action)
    return v, best_action

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    # bookkeeping
    foodList = self.getFoodYouAreDefending(gameState).asList()
    if foodList != self.remainingFoodToDefend:
      # compare self.remainingFoodToDefend and foodList to find out which food was eaten
      self.last_seen += [i for i in self.remainingFoodToDefend if i not in foodList]
    
    # filter out positions in last_seen that you are close enough (<=5) to
    self.last_seen = [i for i in self.last_seen if self.getMazeDistance(gameState.getAgentPosition(self.index), i) > 3]

    # print("last_seen = ", self.last_seen)
    
    # update remaining food to defend
    self.remainingFoodToDefend = foodList

    # # find list of enemy pacs
    # enemy_pac_list = self.getOpponents(gameState)
    # enemy_pac_list = [i for i in enemy_pac_list if gameState.getAgentState(i).isPacman]
    # enemy_pac_pos_list = [gameState.getAgentPosition(i) for i in enemy_pac_list if gameState.getAgentPosition(i) != None]

    # if len(self.prev_enemy_pac_list) == len(enemy_pac_pos_list) + 1:
    #   if self.prev_min_pac_dist == 1:
    #     self.kill_timer = 10

    depth_list = [2, 6, 10, 14] # best IDS Scenario
    # depth_list = [10] # no IDS for now 

    total_compute_time = 0.995 # time left for the agent to choose an action

    depth_act_dict = {}

    # reset dictionaries
    self.max_agent_dict = {}
    self.min_agent_dict = {}
    self.heurvalue_dict = {}


    # iteratively deepening search
    start_time = time.time()

    for depth in depth_list:
      # time bookkeeping      
      current_time = time.time()
      time_spent = current_time - start_time
      time_left = total_compute_time - time_spent

      if time_left < 0.001:
        break
      
      v, best_action = self.max_agent(gameState, self.index, depth, time_left)
      if best_action != None:
        depth_act_dict[depth] = best_action
      else:
        break

    # print("depth_act_dict = ", depth_act_dict)
    if depth_act_dict == {}:
      # print("Time taken by Defense = ", time.time() - start_time)
      print("Defense is Random")
      return random.choice(actions)
    else:
      # choose action with highest depth
      best_depth = max(depth_act_dict.keys())
      best_action = depth_act_dict[best_depth]

      # print time taken for agent to choose action
      # print("Time taken by Defense = ", time.time() - start_time)
      # print("best_depth = ", best_depth)



      # if best action is stop, NEVER ALLOW THIS TO HAPPEN 
    #   if best_action == Directions.STOP :
    #     # print("stop was chosen as best action")
        
    #     # go through all depths and find the first non stop action
    #     for depth in depth_act_dict.keys():
    #         if depth_act_dict[depth] != Directions.STOP:
    #             best_action = depth_act_dict[depth]
    #             break
    #     if best_action == Directions.STOP:
    #         # remove stop from actions
    #         # actions.remove(Directions.STOP)
    #         best_action = random.choice(actions)

    #   print("[defense] total time taken = ", time.time() - start_time)
      
      # if self.kill_timer > 0:
      #   self.kill_timer -= 1

    # enemy_pos_list = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if gameState.getAgentPosition(i) != None]

    # enemy_dist_list = [self.getMazeDistance(gameState.getAgentPosition(self.index), enemy_pos) for enemy_pos in enemy_pos_list]

    # # reset self.prev_enemy_pac_list
    # self.prev_enemy_pac_list = enemy_pos_list
    
    # if len(enemy_dist_list) > 0:
    #   self.prev_min_pac_dist = min(enemy_dist_list)
    # else:
    #   self.prev_min_pac_dist = math.inf



    return best_action    
    











class LinkedList:
  """Class that finds dead ends in the maze and stores them in a linked list"""

  def __init__(self, currentPos:Tuple[int, int], parentPos:Tuple[int, int], legalActions:List[Directions]):
    self.currentPos = currentPos
    self.parentPos = parentPos
    self.legalActions = legalActions
    self.next = None

    self.numbOfLegalActions = len(legalActions)

    # check if dead end: stop and reverse always possible options => if only options then dead end
    self.isDeadEnd = True if self.numbOfLegalActions == 2 else False

    # if dead end, use debug draw to draw red point 
    if self.isDeadEnd:
      CaptureAgent.debugDraw(self.currentPos, [1, 0, 0], clear=False)


