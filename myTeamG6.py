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
from util import nearestPoint
from util import manhattanDistance


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'SwitchAgent', second = 'SwitchAgent'):
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


#################
#               #
# Dynamic Agent # 
#               #
#################

DEPTH = int(15) # adversarial search tree depth

class DynamicAgent(CaptureAgent):
  """
  Parent class for multi agent search agents
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.depth = DEPTH
    CaptureAgent.registerInitialState(self, gameState)

    self.currentMission = ""
    self.currentMissionCounter = 0
    self.pointToGoTo = (0, 0)
    self.walls = gameState.getWalls().asList()
    self.middleX = gameState.data.layout.width // 2
    self.previouselyExistingFood = \
      self.getFoodYouAreDefending(gameState).asList()
    
    # Switch 
    indices = []
    self.enemyIndices = []
    if (gameState.isOnRedTeam(self.index)):
      indices = gameState.getRedTeamIndices()
      self.enemyIndices = gameState.getBlueTeamIndices()
    
    else:
      indices = gameState.getBlueTeamIndices()
      self.enemyIndices = gameState.getRedTeamIndices()

    defensiveIndex = min(indices)
    offensiveIndex = max(indices)

    self.isDefense = self.index==defensiveIndex
    self.isOffense = self.index==offensiveIndex

  def returnToHome(self, gameState):
     '''Go back to home side if carrying certain percentage of pellets'''
     threshold = 0.05 * len(self.getFood(gameState).asList())
     if gameState.getAgentState(self.index).numCarrying >= threshold:
        possibleActions = gameState.getLegalActions(self.index)
        possibleActions = self.removeStopFromActions(possibleActions)
        action_scores = [self.evaluateGoHome(gameState,action) for action in possibleActions]

        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]

        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]
     
     else:
        return None

  def findFood(self, gameState, action):
    successor = self.getSuccessor(gameState,action)
    successor = gameState
    foodList = self.getFood(successor).asList()   

    # Compute distance to the nearest food
    if len(foodList) > 0:
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food \
                         in foodList])
      if minDistance > 0:
         score = 1000* 1.0/minDistance
    else:
       score = 0

    print(score)
    return score
      



  def avoidGhosts(self, gameState, action):
      # Agent's position before and after action
      currentPos = gameState.getAgentPosition(self.index)
      successor = self.getSuccessor(gameState,action)
      successorPos = successor.getAgentPosition(self.index)

     # Ghost positions in current state
      ghostPositions = []
      for enemy in self.enemyIndices:
          agent = gameState.getAgentState(enemy)
          agentPos = gameState.getAgentPosition(enemy)
          if not agent.isPacman and agentPos is not None:
            ghostPositions.append(agentPos)
 
       # Compare pacmans distance to ghosts
      if ghostPositions:
         nearestGhost = min(ghostPositions)
         currentDist = self.getMazeDistance(currentPos,nearestGhost)
         successorDist = self.getMazeDistance(successorPos,nearestGhost)
         score = successorDist-currentDist
      else:
         score = float('inf')
      
      return score

  
  def evaluateOffense(self, gameState, action):
    """
    This function scores the given state of the game for an offensive Pac-Man agent.
    """

    # define feature weights
    foodWeight = 0.0
    capsuleWeight = 0.0
    ghostWeight = 10.0
    scoreWeight = 0.0

    # get current state information
    successor = self.getSuccessor(gameState,action)
    currentPos = gameState.getAgentPosition(self.index)
    successorPos = successor.getAgentPosition(self.index)
    currentFood = self.getFood(gameState).asList()
    currentCapsules = self.getCapsules(gameState)
    currentScore = self.getScore(gameState)

    # calculate distance to nearest food pellet
    foodDistances = [self.getMazeDistance(successorPos, food) for food in \
                      currentFood]
    if foodDistances:
        nearestFood = min(foodDistances)
    else:
        nearestFood = 0

    # calculate distance to nearest ghost
    ghostDistances = []
    for enemy in self.enemyIndices:
        agent = gameState.getAgentState(enemy)
        agentPos = gameState.getAgentPosition(enemy)
        if not agent.isPacman and agentPos is not None:
          ghostDistances.append(self.getMazeDistance(successorPos, agentPos))
    
    if ghostDistances:
        nearestGhost = min(ghostDistances)
    else:
        nearestGhost = 0

    # calculate score for eating food
    foodScore = 0
    if nearestFood:
        foodScore = 1.0 / nearestFood

    # calculate score for capsules
    capsuleScore = 0
    if currentCapsules:
        nearestCapsule = min([self.getMazeDistance(currentPos, capsule) \
                              for capsule in currentCapsules])
        capsuleScore = 1.0 / nearestCapsule

    # calculate score for enemy ghosts
    ghostScore = 0
    if nearestGhost:
        ghostScore = -1.0 / nearestGhost+1.0

    # calculate score for current score
    scoreScore = currentScore

    # calculate total score
    totalScore = foodWeight * foodScore + capsuleWeight * capsuleScore + ghostWeight * ghostScore + scoreWeight * scoreScore

    return totalScore


  def evaluateGoHome(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    successor = self.getSuccessor(gameState,action)
    pacmanPosition = successor.getAgentPosition(self.index)
    
    # Calculate distance to nearest ghost
    ghostScore = 0
    ghostDistances = []
    for agent in self.enemyIndices:
        agentPos = successor.getAgentPosition(agent)
        if agentPos is not None and not successor.getAgentState(agent).isPacman:
            ghostDistances.append(self.getMazeDistance(pacmanPosition, agentPos))

    for ghost in ghostDistances:
        if ghost <= 1.0:
            ghostScore = float('inf')
        else:
            ghostScore += 1.0 / ghost

    # Calculate distance home
    homeScore = 0
    homeDistance = self.getMazeDistance(pacmanPosition,self.start)
    if homeDistance > 0:
       homeScore = 1.0/homeDistance

    score = homeScore - ghostScore

    return score

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    currPos = gameState.getAgentPosition(self.index)
    successor = self.getSuccessor(gameState,action)
    newPos = successor.getAgentPosition(self.index)
    penalty = self.punishBackAndForth(self.index, newPos, currPos, 0)

    # Calculate distance to nearest ghost
    ghostScore = 0
    ghostDistances = []
    for agent in self.enemyIndices:
        agentPos = successor.getAgentPosition(agent)
        if agentPos is not None and not successor.getAgentState(agent).isPacman:
            ghostDistances.append(self.getMazeDistance(newPos, agentPos))

    for ghost in ghostDistances:
        if ghost <= 1.0:
            ghostScore = 1000
        else:
            ghostScore += 1.0 / ghost

    return features * weights - 500*ghostScore
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': 10000}
  
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
    
  def punishBackAndForth(self, agent, position, prev_position, penalty):
    if agent == self.index:
        if position == prev_position:
            return penalty
    return 0
    
  def removeStopFromActions(self, actions):
      if 'Stop' in actions:
        actions.remove('Stop')
      return actions
  
  def bestActionToGetToPoint(self, gameState, point):
      bestDistance = float('inf')
      bestAction = ""
      for action in gameState.getLegalActions(self.index):
        d = self.distancer.getDistance(gameState.generateSuccessor(self.index, action).getAgentState(self.index).getPosition(), point)
        if d < bestDistance:
          bestDistance = d
          bestAction = action

      return bestAction

              
  
################
#              #
# Switch Agent # 
#              #
################

# inherets from Dynamic Agent but can switch between defensive/offensive
class SwitchAgent(DynamicAgent):

  def chooseAction(self, gameState):
    # Check offense or defense
    if (self.isDefense):
      return self.chooseActionDefensiveBehaviour(gameState)
    
    else:
      # Expectimax/Alpha-beta offense for pacman
      if (gameState.getAgentState(self.index).isPacman):
        action = self.returnToHome(gameState)
        if action is not None:
           return action
        else:
           return self.chooseActionAlphaBeta(gameState)

      # Reflex offense for ghost
      else:
        return self.chooseActionReflex(gameState)
      
      
  #####################
  #                   #
  # ALPHA-BETA ACTION # 
  #                   #
  #####################

  def chooseActionAlphaBeta(self, gameState):
    
    """
    Chooses pacman action based on alpha-beta pruning expectimax with DEPTH
    """
    possibleActions = gameState.getLegalActions(self.index)
    possibleActions = self.removeStopFromActions(possibleActions)
    action_scores = [self.alpha_beta(0, 0, self.getSuccessor(gameState,action),float('inf'),-float('inf')) for action in possibleActions]

    max_action = max(action_scores)
    max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]

    chosenIndex = random.choice(max_indices)

    return possibleActions[chosenIndex]
  
  def alpha_beta(self, agent, depth, gameState, alpha, beta):
    if gameState.isOver() or depth == self.depth:
        actions = gameState.getLegalActions(self.index)
        actions = self.removeStopFromActions(actions)
        max_score = -float('inf')
        for action in actions:
            score = self.evaluate(gameState, action)
            max_score = max(max_score, score)
        return max_score

    if agent == self.index:  # maximize for our team
        actions = gameState.getLegalActions(agent)
        actions = self.removeStopFromActions(actions)
        max_score = -float('inf')
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.alpha_beta((agent + 1) % gameState.getNumAgents(), depth, successor, alpha, beta)
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)
            if beta <= alpha:
                break  # beta cut-off
        return max_score

    else:  # minimize for other team
        if (not gameState.getAgentState(agent).isPacman) and (gameState.getAgentPosition(agent) is tuple):  # minimize for ghosts
            actions = gameState.getLegalActions(agent)
            actions = self.removeStopFromActions(actions)
            min_score = float('inf')
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                score = self.alpha_beta((agent + 1) % gameState.getNumAgents(), depth, successor, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break  # alpha cut-off
            return min_score

        else:  # ignore Pacman agents
            if gameState.getAgentPosition(agent) is tuple:
                actions = gameState.getLegalActions(agent)
                actions = self.removeStopFromActions(actions)
                avg_score = 0
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    score = self.alpha_beta((agent + 1) % gameState.getNumAgents(), depth, successor, alpha, beta)
                    avg_score += score
                return avg_score / len(actions)
            
            else:  # ghost is scared or has just died
                actions = gameState.getLegalActions(self.index)
                actions = self.removeStopFromActions(actions)
                max_score = -float('inf')
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    score = self.alpha_beta((self.index + 1) % gameState.getNumAgents(), depth+1, successor, alpha, beta)
                    max_score = max(max_score, score)
                    alpha = max(alpha, max_score)
                    if beta <= alpha:
                        break  # beta cut-off
                return max_score


  #####################
  #                   #
  # EXPECTIMAX ACTION #
  #                   #
  #####################
  def chooseActionExpectimax(self, gameState):
    
    """
    Chooses pacman action based on expectimaz with DEPTH
    """
    possibleActions = gameState.getLegalActions(self.index)
    action_scores = [self.expectimax(0, 0, self.getSuccessor(gameState,action)) for action in possibleActions]

    max_action = max(action_scores)
    max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]

    chosenIndex = random.choice(max_indices)

    return possibleActions[chosenIndex]

  def expectimax(self, agent, depth, gameState):
    if gameState.isOver() or depth == self.depth:
        actions = gameState.getLegalActions(self.index)
        max_score = -float('inf')
        for action in actions:
            score = self.evaluate(gameState,action)
            max_score = max(max_score, score)

        return max_score

    if agent == self.index:  # maximize for our team
        actions = gameState.getLegalActions(agent)
        max_score = -float('inf')
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.expectimax((agent + 1) % gameState.getNumAgents(), depth, successor)
            max_score = max(max_score, score)

        return max_score

    else:  # minimize for other team
        if (not gameState.getAgentState(agent).isPacman) and (gameState.getAgentPosition(agent) is tuple):  # minimize for ghosts
            actions = gameState.getLegalActions(agent)
            num_actions = len(actions)
            min_score = float('inf')
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                score = self.expectimax((agent + 1) % gameState.getNumAgents(), depth, successor)
                min_score = min(min_score, score)

            return min_score

        else:  # ignore Pacman agents
            if gameState.getAgentPosition(agent) is tuple:
              avg_score = 0
              for action in actions:
                  successor = self.getSuccessor(gameState, action)
                  score = self.expectimax((agent + 1) \
                                          % gameState.getNumAgents(), depth,  successor)
                  avg_score += score

              return avg_score / num_actions
            
            else: 
                actions = gameState.getLegalActions(self.index)
                max_score = -float('inf')
                for action in actions:
                    successor = self.getSuccessor(gameState, action)
                    score = self.expectimax((self.index + 1) % gameState.getNumAgents(),depth+1, successor)
                    max_score = max(max_score, score)

                return max_score
            

  
  #################
  #               #
  # REFLEX ACTION #
  #               #
  #################
  def chooseActionReflex(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluateReflex(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)
  
  def evaluateReflex(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
  
  
  ###########################
  #                         #
  # DEFENSIVE BAHVIOUR TREE # 
  #                         #
  ###########################

  def patrolFood(self, gameState):
      if self.currentMission != "GUARD FOOD":
        self.currentMission = "GUARD FOOD"
        self.currentMissionCounter = 0

      existingFoodPositions = self.getFoodYouAreDefending(gameState).asList()
      existingFoodPositions.sort(key=lambda x: abs(x[0] - self.middleX))

      self.pointToGoTo = existingFoodPositions[0]
      if len(existingFoodPositions) > 0 and gameState.getAgentPosition(self.index) == self.pointToGoTo:
        self.pointToGoTo = existingFoodPositions[1]


  def chooseActionDefensiveBehaviour(self, gameState):
      startTime = time.time()

      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      invaders = [enemy.getPosition() for enemy in enemies if enemy.isPacman]
      seenInvaders = [enemy.getPosition() for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
      foodEaten = list(set(self.previouselyExistingFood) - set(self.getFoodYouAreDefending(gameState).asList()))


      """ ------------------- Behavior Tree ------------------- """

      if len(invaders) != 0:
        if len(seenInvaders) != 0: # Invader exists, and is visible -> chase invader
          if self.currentMission != "CHASE":
            self.currentMission = "CHASE"
            self.currentMissionCounter = 0
          self.pointToGoTo = invaders[0]
        
        else: 
          if len(foodEaten) != 0: # Invader exists, is not visible, but food has been eaten -> go to eaten food
            if self.currentMission != "GO TO EATEN FOOD":
              self.currentMission = "GO TO EATEN FOOD"
              self.currentMissionCounter = 0
            self.pointToGoTo = foodEaten[0]

          else:
            if len(self.getCapsulesYouAreDefending(gameState)) != 0: # Invader exists, is not visible -> try go guard capsule
              if self.currentMission != "GUARD CAPSULE":
                self.currentMission = "GUARD CAPSULE"
                self.currentMissionCounter = 0
              self.pointToGoTo = self.getCapsulesYouAreDefending(gameState)[0]


            elif self.currentMission != "GUARD FOOD" or  gameState.getAgentPosition(self.index) == self.pointToGoTo: # Invader exists, is not visible, capsule don't exist -> guard food
              self.patrolFood(gameState)

      elif self.currentMission != "GUARD FOOD" or  gameState.getAgentPosition(self.index) == self.pointToGoTo:
        self.patrolFood(gameState)

      """ ----------------------------------------------------- """

      while time.time() - startTime < 0.1:
        pass

      self.currentMissionCounter += 1
      self.previouselyExistingFood = self.getFoodYouAreDefending(gameState).asList()

      return self.bestActionToGetToPoint(gameState, self.pointToGoTo)

