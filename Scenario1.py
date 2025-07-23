from FourRooms import FourRooms
import numpy as np
import random
import sys

def main():

    #get input from user to see if stochastic or not
    stochastic = False
    if(len(sys.argv) > 1):
        if(sys.argv[1] == '-stochastic'):
            stochastic = True
            print("Using stochastic environment:")
        else:
            print("Using deterministic environment")

    #create FourRooms Object
    fourRoomsObj = FourRooms('simple', stochastic)

    #dimensions
    rows = 13
    cols = 13

    #convert coordinates to an array index
    def convert(pos):
        return pos[0] + cols * pos[1]
    
    #initialize Q-values for each state-action pair
    Q_values = np.zeros((rows*cols, 4))

    #actions
    actionList = [fourRoomsObj.UP, fourRoomsObj.DOWN, fourRoomsObj.LEFT, fourRoomsObj.RIGHT]

    #hyperparameters
    epsilon = 0.2
    discount_factor = 0.9
    learning_rate = 0.1

    #track stpes per episode and total reward per episode for performance testing
    episode_steps = []  # Track steps per episode
    total_rewards = []  # Track total reward per episode

    #values for learning epochs
    episodes = 1000
    numMoves = 0
    totalReward = 0

    #training loop
    for episode in range(episodes):
        #new training epoch each loop
        fourRoomsObj.newEpoch()

        #make sure simulation still running
        while(fourRoomsObj.isTerminal() == False):
            #increment moves
            numMoves = numMoves + 1

            #state before action
            oldState = fourRoomsObj.getPosition()

            #choose next action

            ########
            ##random action selection method
            ########
            #action = random.randint(0, 3)

            ########
            ##Epsilon-greedy action selection method
            ########
            if random.random() < epsilon:
                #explore --> choose a random action
                action = random.randint(0, 3)
            else:
                #exploit --> choose the best action
                action = np.argmax(Q_values[convert(oldState), :])

            #take the action
            cellType, currentState, packageNum, isTerm = fourRoomsObj.takeAction(actionList[action])

            #reward functions:
            #a package
            if(cellType == 1 or cellType == 2 or cellType == 3):
                reward = 100
            #tried to move to an out of bounds area
            elif(oldState == currentState):
                reward = -100
            #not a package or movement wasn't useful
            else:
                reward = -1

            totalReward = totalReward + reward

            #update Q values using the Q-learning update rule
            Q_values[convert(oldState), action] = Q_values[convert(oldState), action] + learning_rate * (
                reward + discount_factor * np.max(Q_values[convert(currentState), :]) - 
                Q_values[convert(oldState), action]
            )

        #store episode metrics
        episode_steps.append(numMoves)
        total_rewards.append(totalReward)


        #print out  info about episode (only every 100)
        if episode % 100 == 0:
            print(f"Episode {episode}: Steps = {numMoves}, Total Reward = {totalReward:.2f}")

        #reset episode metrics
        numMoves = 0
        totalReward = 0

    #print path taken
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()