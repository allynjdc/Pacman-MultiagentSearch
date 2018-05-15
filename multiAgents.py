# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # HINTS:
        # Given currentGameState and successorGameState, determine if the next state is good / bad
        # Compute a numerical score for next state that will reflect this
        # Base score = successorGameState.getScore() - Line 77
        # Can increase / decrease this score depending on:
        #   new pacman position, ghost position, food position, 
        #   distances to ghosts, distances to food
        # You can choose which features to use in your evaluation function
        # You can also put more weight to some features

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        action,score = self.value(gameState, currentAgentIndex, currentDepth)
        return action 

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth):
        # pass
        # Check when to update depth
        # check if currentDepth == self.depth
        #   if it is, stop recursion and return score of gameState based on self.evaluationFunction
        # check if gameState.isWin() or gameState.isLose()
        #   if it is, stop recursion and return score of gameState based on self.evaluationFunction
        # check whether currentAgentIndex is our pacman agent or ghost agent
        #   if our agent: return max_value(....)
        #   otherwise: return min_value(....)

        ######### CODE #########

        # CHECKING THE CURRENT AGENT'S INDEX, 
        #   IF IT EXCEEDS TO THE TOTAL NUMBER OF AGENTS, 
        #   THEN WE'LL UPDATE THE DEPTH VALUE AND SET THE INDEX BACK TO ZERO (OUR AGENT -- PACMAN)
        #   OTHERWISE: IGNORE IT.
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            currentDepth += 1

        # IF THE CURRENT DEPTH VALUE IS EQUAL TO THE SELF.DEPTH VALUE,
        #   WILL RETURN THE SCORE OF GAMESTATE BASED ON SELF.EVALUATIONFUNCTION
        #   OTHERWISE: IGNORE IT.
        # EQUIVALENT TO: if cutoff(state,depth): evaluation_function(state)
        if currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        # IF THE GAMESTATE IS NOW WINNING OR LOSING,
        #   WILL RETURN THE SCORE OF GAMESTATE BASED ON SELF.EVALUATIONFUNCTION
        #   OTHERWISE: IGNORE IT.
        # EQUIVALENT TO: if state is terminal state: value(state)
        if gameState.isWin() or gameState.isLose():
            # return self.evaluationFunction(gameState)
            return gameState.getScore()

        # CHECKING THE CURRENT AGENT'S INDEX,
        #   IF THE CURRENT AGENT IS EQUAL TO ZERO (OUR AGENT -- PACMAN),
        #   THEN WE'LL GET THE MAXIMUM VALUE FROM THE FOLLOWING SUCCESSORS.
        #   OTHERWISE: IGNORE IT.
        # EQUIVALENT TO: if state is max-node: max-value(state,depth)
        if currentAgentIndex == 0:
            return self.max_value(gameState, currentAgentIndex, currentDepth)
        
        # WILL BE USED IF THE CURRENT AGENT'S INDEX IS NOT EQUAL TO ZERO (OUR AGENT -- PACMAN)
        #   WE'LL GET THE MINIMUM VALUE FROM THE FOLLOWING SUCCESSORS.
        # EQUIVALENT TO: if state is min-node: min-value(state,depth)
        return self.min_value(gameState, currentAgentIndex, currentDepth)


    # Note: always returns (action,score) pair
    def max_value(self, gameState, currentAgentIndex, currentDepth):
        # pass
        # current_value = -inf
        # loop over each action available to current agent:
        # (hint: use gameState.getLegalActions(...) for this)
        #     use gameState.generateSuccessor to get nextGameState from action
        #     compute value of nextGameState by calling self.value
        #     compare value of nextGameState and current_value
        #     keep whichever value is bigger, and take note of the action too
        # return (action,current_value)

        ######### CODE #########

        # SETTING THE CURRENT VALUE INTO A NEGATIVE INFINITY, AND
        # SETTING THE TEMPORARY RETURN VALUE AS A TUPLE WHICH 
        #   CONSISTS AN ACTION AND ITS CURRENT VALUE
        # EQUIVALENT TO: v = -inf
        current_value = float('inf') * -1
        act = ("unknown", current_value)

        # LOOPING OVER EACH ACTION AVAILABLE TO CURRENT AGENT
        # IN EACH LOOP:
        #   - WILL GET THE NEXT VALUE OF THE NEXT STATE BY CALLING THE FUNCTION 
        #     VALUE() BY INCREMENTING THE CURRENT AGENT INDEX INTO 1.
        #   - WILL GET THE BIGGER VALUE OUT OF THE TWO VALUES, 
        #     NEXT STATE'S VALUE OR TEMPORARY VALUE.
        #   - IF THE NEXT STATE'S VALUE IS THE BIGGER VALUE, 
        #     THEN WE'LL UPDATE THE VALUES OF TUPLE, ACTION AND THE TEMPORARY VALUE.
        # EQUIVALENT TO: 
        #   for action,next_state in successors(state):
        #       next_v = minimax-value(next_state,depth+1)
        #       v = max(v,next_v)
        for action in gameState.getLegalActions(currentAgentIndex):
            next_v = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth)
            current_value = max(act[1], next_v[1] if type(next_v) is tuple else next_v)
            if current_value is not act[1]:
                act = (action, current_value)

        # WILL SIMPLY RETURN THE TUPLE WHICH CONSISTS
        # THE ACTION OF THE STATE'S MAXIMUM VALUE AND ITS VALUE.
        # EQUIVALENT TO:  return v 
        return act


    # Note: always returns (action,score) pair
    def min_value(self, gameState, currentAgentIndex, currentDepth):
        # pass
        # current_value = inf
        # loop over each action available to current agent:
        # (hint: use gameState.getLegalActions(...) for this)
        #     use gameState.generateSuccessor to get nextGameState from action
        #     compute value of nextGameState by calling self.value
        #     compare value of nextGameState and current_value
        #     keep whichever value is smaller, and take note of the action too
        # return (action,current_value)

        ######### CODE #########

        # SETTING THE CURRENT VALUE INTO A NEGATIVE INFINITY, AND
        # SETTING THE TEMPORARY RETURN VALUE AS A TUPLE WHICH 
        #   CONSISTS AN ACTION AND ITS CURRENT VALUE
        # EQUIVALENT TO: v = -inf
        current_value = float('inf')
        act = ("None", current_value)

        # LOOPING OVER EACH ACTION AVAILABLE TO CURRENT AGENT
        # IN EACH LOOP:
        #   - WILL GET THE NEXT VALUE OF THE NEXT STATE BY CALLING THE FUNCTION 
        #     VALUE() BY INCREMENTING THE CURRENT AGENT INDEX INTO 1.
        #   - WILL GET THE BIGGER VALUE OUT OF THE TWO VALUES, 
        #     NEXT STATE'S VALUE OR TEMPORARY VALUE.
        #   - IF THE NEXT STATE'S VALUE IS THE LESSER VALUE, 
        #     THEN WE'LL UPDATE THE VALUES OF TUPLE, ACTION AND THE TEMPORARY VALUE.
        # EQUIVALENT TO: 
        #   for action,next_state in successors(state):
        #       next_v = minimax-value(next_state,depth+1)
        #       v = min(v,next_v)
        for action in gameState.getLegalActions(currentAgentIndex):
            next_v = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth)
            current_value = min(act[1], next_v[1] if type(next_v) is tuple else next_v)
            if current_value is not act[1]:
                act = (action, current_value)

        # WILL SIMPLY RETURN THE TUPLE WHICH CONSISTS
        # THE ACTION OF THE STATE'S MAXIMUM VALUE AND ITS VALUE.
        # EQUIVALENT TO:  return v 
        return act



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        alpha = float('inf') * -1
        beta = float('inf')
        action,score = self.value(gameState, currentAgentIndex, currentDepth, alpha, beta)
        return action 

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
        # pass
        # More or less the same with MinimaxAgent's value() method
        # Just update the calls to max_value and min_value (should now include alpha, beta params)

        ######### CODE #########

        # CHECKING THE CURRENT AGENT'S INDEX, 
        #   IF IT EXCEEDS TO THE TOTAL NUMBER OF AGENTS, 
        #   THEN WE'LL UPDATE THE DEPTH VALUE AND SET THE INDEX BACK TO ZERO (OUR AGENT -- PACMAN)
        #   OTHERWISE: IGNORE IT.
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            currentDepth += 1

        # IF THE CURRENT DEPTH VALUE IS EQUAL TO THE SELF.DEPTH VALUE,
        #   WILL RETURN THE SCORE OF GAMESTATE BASED ON SELF.EVALUATIONFUNCTION
        #   OTHERWISE: IGNORE IT.
        # EQUIVALENT TO: if cutoff(state,depth): evaluation_function(state)
        if currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        # IF THE GAMESTATE IS NOW WINNING OR LOSING,
        #   WILL RETURN THE SCORE OF GAMESTATE BASED ON SELF.EVALUATIONFUNCTION
        #   OTHERWISE: IGNORE IT.
        # EQUIVALENT TO: if state is terminal state: value(state)
        if gameState.isWin() or gameState.isLose():
            # return self.evaluationFunction(gameState)
            return gameState.getScore()

        # CHECKING THE CURRENT AGENT'S INDEX,
        #   IF THE CURRENT AGENT IS EQUAL TO ZERO (OUR AGENT -- PACMAN),
        #   THEN WE'LL GET THE MAXIMUM VALUE FROM THE FOLLOWING SUCCESSORS.
        #   OTHERWISE: IGNORE IT.
        # EQUIVALENT TO: if state is max-node: max-value(state,depth)
        if currentAgentIndex == 0:
            return self.max_value(gameState, currentAgentIndex, currentDepth, alpha, beta)
        
        # WILL BE USED IF THE CURRENT AGENT'S INDEX IS NOT EQUAL TO ZERO (OUR AGENT -- PACMAN)
        #   WE'LL GET THE MINIMUM VALUE FROM THE FOLLOWING SUCCESSORS.
        # EQUIVALENT TO: if state is min-node: min-value(state,depth)
        return self.min_value(gameState, currentAgentIndex, currentDepth, alpha, beta)


    # Note: always returns (action,score) pair
    def max_value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
        # pass
        # Similar to MinimaxAgent's max_value() method
        # Include checking if current_value is worse than beta
        #   if so, immediately return current (action,current_value) tuple
        # Include updating of alpha

        ######### CODE #########

        # SETTING THE CURRENT VALUE INTO A NEGATIVE INFINITY, AND
        # SETTING THE TEMPORARY RETURN VALUE AS A TUPLE WHICH 
        #   CONSISTS AN ACTION AND ITS CURRENT VALUE
        # EQUIVALENT TO: v = -inf
        current_value = float('inf') * -1
        act = ("unknown", current_value)

        # LOOPING OVER EACH ACTION AVAILABLE TO CURRENT AGENT
        # IN EACH LOOP:
        #   - WILL GET THE NEXT VALUE OF THE NEXT STATE BY CALLING THE FUNCTION 
        #     VALUE() BY INCREMENTING THE CURRENT AGENT INDEX INTO 1.
        #   - WILL GET THE BIGGER VALUE OUT OF THE TWO VALUES, 
        #     NEXT STATE'S VALUE OR TEMPORARY VALUE.
        #   - IF THE NEXT STATE'S VALUE IS THE BIGGER VALUE, 
        #     THEN WE'LL UPDATE THE VALUES OF TUPLE, ACTION AND THE TEMPORARY VALUE.
        #   - IF THE CURRENT VALUE IS BIGGER TO THE VALUE OF BETA,
        #     THEN WE'LL RETURN THE TUPLE WHICH CONSISTS OF THE ACTION OF THE NEXT STATE'S MAXIMUM VALUE AND ITS VALUE.
        #     THE REMAINING NEXT STATE/S WILL BE PRUNED.
        #   - UPDATE THE VALUE OF ALPHA, BY GETTING THE BIGGER VALUE BETWEEN ALPHA AND NEXT STATE'S VALUE.
        # EQUIVALENT TO: 
        #   for action,next_state in successors(state):
        #       next_v = minimax-value(next_state,depth+1)
        #       v = max(v,next_v)
        #       if v > beta: return v
        #       alpha = max(alpha,v)
        for action in gameState.getLegalActions(currentAgentIndex):
            next_v = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth, alpha, beta)
            current_value = max(act[1], next_v[1] if type(next_v) is tuple else next_v)
            if current_value is not act[1]:
                act = (action, current_value) 
            if current_value > beta:
                return act
            alpha = max(alpha, current_value)

        # WILL SIMPLY RETURN THE TUPLE WHICH CONSISTS
        # THE ACTION OF THE STATE'S MAXIMUM VALUE AND ITS VALUE.
        # EQUIVALENT TO:  return v 
        return act


    # Note: always returns (action,score) pair
    def min_value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
        # pass
        # Similar to MinimaxAgent's min_value() method
        # Include checking if current_value is worse than alpha
        #   if so, immediately return current (action,current_value) tuple
        # Include updating of beta

        ######### CODE #########

        # SETTING THE CURRENT VALUE INTO A NEGATIVE INFINITY, AND
        # SETTING THE TEMPORARY RETURN VALUE AS A TUPLE WHICH 
        #   CONSISTS AN ACTION AND ITS CURRENT VALUE
        # EQUIVALENT TO: v = -inf
        current_value = float('inf')
        act = ("None", current_value)

        # LOOPING OVER EACH ACTION AVAILABLE TO CURRENT AGENT
        # IN EACH LOOP:
        #   - WILL GET THE NEXT VALUE OF THE NEXT STATE BY CALLING THE FUNCTION 
        #     VALUE() BY INCREMENTING THE CURRENT AGENT INDEX INTO 1.
        #   - WILL GET THE BIGGER VALUE OUT OF THE TWO VALUES, 
        #     NEXT STATE'S VALUE OR TEMPORARY VALUE.
        #   - IF THE NEXT STATE'S VALUE IS THE LESSER VALUE, 
        #     THEN WE'LL UPDATE THE VALUES OF TUPLE, ACTION AND THE TEMPORARY VALUE.
        #   - IF THE CURRENT VALUE IS LESSER TO THE VALUE OF ALPHA,
        #     THEN WE'LL RETURN THE TUPLE WHICH CONSISTS OF THE ACTION OF THE NEXT STATE'S MAXIMUM VALUE AND ITS VALUE.
        #     THE REMAINING NEXT STATE/S WILL BE PRUNED.
        #   - UPDATE THE VALUE OF BETA, BY GETTING THE LESSER VALUE BETWEEN BETA AND NEXT STATE'S VALUE.
        # EQUIVALENT TO: 
        #   for action,next_state in successors(state):
        #       next_v = minimax-value(next_state,depth+1)
        #       v = min(v,next_v)
        #       if v < alpha return v
        #       beta = min(beta, v)
        for action in gameState.getLegalActions(currentAgentIndex):
            next_v = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth, alpha, beta)
            current_value = min(act[1], next_v[1] if type(next_v) is tuple else next_v)
            if current_value is not act[1]:
                act = (action, current_value) 
            if current_value < alpha:
                return act
            beta = min(beta, current_value)

        # WILL SIMPLY RETURN THE TUPLE WHICH CONSISTS
        # THE ACTION OF THE STATE'S MAXIMUM VALUE AND ITS VALUE.
        # EQUIVALENT TO:  return v 
        return act


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        action,score = self.value(gameState, currentAgentIndex, currentDepth)
        return action 

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth):
      pass
      # More or less the same with MinimaxAgent's value() method
      # Only difference: use exp_value instead of min_value

    # Note: always returns (action,score) pair
    def max_value(self, gameState, currentAgentIndex, currentDepth):
      pass
      # Exactly like MinimaxAgent's max_value() method

    # Note: always returns (action,score) pair
    def exp_value(self, gameState, currentAgentIndex, currentDepth):
      pass
      # use gameState.getLegalActions(...) to get list of actions
      # assume uniform probability of possible actions
      # compute probabilities of each action
      # be careful with division by zero
      # Compute the total expected value by:
      #   checking all actions
      #   for each action, compute the score the nextGameState will get
      #   multiply score by probability
      # Return (None,total_expected_value) 
      # None action --> we only need to compute exp_value but since the 
      # signature return values of these functions are (action,score), we will return an empty action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    # Similar to Q1, only this time there's only one state (no nextGameState to compare it to)
    # Use similar features here: position, food, ghosts, scared ghosts, distances, etc.
    # Can use manhattanDistance() function
    # You can add weights to these features
    # Update the score variable (add / subtract), depending on the features and their weights
    # Note: Edit the Description in the string above to describe what you did here

    return score

# Abbreviation
better = betterEvaluationFunction

