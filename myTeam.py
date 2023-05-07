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
import math

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

  # evaluation function
  def evaluationFunction(self, gameState):
    # heuristic function to evaluate the state of the game

    # if terminal state    
    if gameState.isOver():
      return gameState.getScore()
    
    if False:
      # if pacman is eaten: seen by spawning at the start
      if gameState.getAgentState(self.index).isPacman and gameState.getAgentState(self.index).configuration.pos == gameState.getInitialAgentPosition(self.index):
        return -math.inf
      
      # if pacman eats a food
      if gameState.getAgentState(self.index).numCarrying > 0:
        return math.inf
      
      # if pacman is close to a food
      foodList = self.getFood(gameState).asList()
      minFoodDistance = min([self.getMazeDistance(gameState.getAgentPosition(self.index), food) for food in foodList])
      if minFoodDistance < 3:
        return 1/minFoodDistance
      
      # if pacman is close to a ghost
      ghostList = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if not gameState.getAgentState(i).isPacman]
      # filter out ghosts that are not visible
      ghostList = [ghost for ghost in ghostList if ghost!=None]

      if any(ghostList):
        minGhostDistance = min([self.getMazeDistance(gameState.getAgentPosition(self.index), ghost) for ghost in ghostList])
        if minGhostDistance < 3:
          return -1/minGhostDistance
      
      # if pacman is close to a capsule
      capsuleList = self.getCapsules(gameState)
      minCapsuleDistance = min([self.getMazeDistance(gameState.getAgentPosition(self.index), capsule) for capsule in capsuleList])
      if minCapsuleDistance < 3:
        return 1/minCapsuleDistance
      
      return 0
      
    else:
      # new heuristic function
      # list of our food, enemy food, enemy pacman, enemy ghost, our pacman, our ghost, capsule
      foodList = self.getFood(gameState).asList()
      enemyFoodList = self.getFoodYouAreDefending(gameState).asList()
      enemyPacmanList = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman]
      enemyGhostList = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if not gameState.getAgentState(i).isPacman]
      enemyGhostList = [ghost for ghost in enemyGhostList if ghost!=None]
      ourPacmanList = [gameState.getAgentPosition(i) for i in self.getTeam(gameState) if gameState.getAgentState(i).isPacman]
      ourGhostList = [gameState.getAgentPosition(i) for i in self.getTeam(gameState) if not gameState.getAgentState(i).isPacman]
      ourGhostList = [ghost for ghost in ourGhostList if ghost!=None]
      capsuleList = self.getCapsules(gameState)

      # if terminal state
      if gameState.isOver():
        return gameState.getScore()
      
      
      


      return 0

    

  # setup for minimax
  def getSuccessors(self, gameState):
    """
    Returns successor states, the actions they require, and a cost of 1.

    The following successor states are the same board, but the agent
    has been moved one step in the specified direction.

    """
    successors = []
    for action in gameState.getLegalActions(self.index):
      # Ignore illegal actions
      if action == Directions.STOP:
        continue
      successor = gameState.generateSuccessor(self.index, action)
      successors.append((successor, action))
    return successors
  
  def max_agent(self, gameState, depth, time_left = math.inf, alpha = -math.inf, beta = math.inf):
    if depth == 0 or time_left < 0.05:
      return self.evaluationFunction(gameState)
    v = -math.inf
    for successor, action in self.getSuccessors(gameState):

      # check if enemy is visible from the successor state
      enemyList = [successor.getAgentPosition(i) for i in self.getOpponents(successor)]
      if any(enemyList):
        v = max(v, self.min_agent(successor, depth-1, time_left - 0.05, alpha, beta))
      
      else:
        # if enemy is not visible, then assume board remains the same and make next move
        v = max(v, self.max_agent(successor, depth - 1, time_left - 0.05, alpha, beta))

      if v >= beta:
        return v
      alpha = max(alpha, v)
    return v
  
  def min_agent(self, gameState, depth, time_left = math.inf, alpha = -math.inf, beta = math.inf):
    if depth == 0 or time_left < 0.05:
      return self.evaluationFunction(gameState)
    v = math.inf
    for successor, action in self.getSuccessors(gameState):
        
        # check if enemy is visible from the successor state
        enemyList = [successor.getAgentPosition(i) for i in self.getOpponents(successor)]
        if any(enemyList):
          v = min(v, self.max_agent(successor, depth-1, time_left - 0.05, alpha, beta))
        
        else:
          # if enemy is not visible, then assume board remains the same and make next move
          v = min(v, self.min_agent(successor, depth - 1, time_left - 0.05, alpha, beta))
  
        if v <= alpha:
          return v
        beta = min(beta, v)
    return v
  
  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    # minimax
    depth = 5
    time_left = 0.95 # time left for the agent to choose an action
    best_action = None
    best_score = -math.inf
    for successor, action in self.getSuccessors(gameState):
      score = self.min_agent(successor, depth, time_left - 0.05)
      if score > best_score:
        best_score = score
        best_action = action
    return best_action
  
    return random.choice(actions)

