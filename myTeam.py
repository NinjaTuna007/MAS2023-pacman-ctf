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
               first = 'Terminator5000', second = 'Terminator5000'):
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
    # if self.teamRed: # i am red
    #   self.captureFood = gameState.getBlueFood() # food on the blue team's side
    #   self.defendFood = gameState.getRedFood() # food on the red team's side
    
    # else: # i am blue
    #   self.captureFood = gameState.getRedFood()
    #   self.defendFood = gameState.getBlueFood()






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



























