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
    fourRoomsObj = FourRooms('rgb', stochastic)

    #dimensions
    rows = 13
    cols = 13

    #convert coordinates to an array index
    #includes number of packages left -> unique state for each combo of position and package num left
    def convert(pos, packagesLeft):
        return pos[0] + cols * pos[1] + rows * cols * (3 - packagesLeft)
    
    #initialize Q-values for each state-action pair
    #include for each possible num of packages remaining
    Q_values = np.zeros((rows * cols * (3 + 1), 4))

    #requried order of collection - red, green, blue
    requiredOrder = [1, 2, 3]

    #actions
    actionList = [fourRoomsObj.UP, fourRoomsObj.DOWN, fourRoomsObj.LEFT, fourRoomsObj.RIGHT]

    #hyperparameters
    epsilon = 0.3
    discount_factor = 0.95
    learning_rate = 0.2

    #track stpes per episode and total reward per episode for performance testing
    episode_steps = []  # Track steps per episode
    total_rewards = []  # Track total reward per episode

    #values for learning epochs
    episodes = 3000
    numMoves = 0
    totalReward = 0

    #values to track best episode
    bestEpisode = 0
    bestSteps = float('inf')

    #training loop
    for episode in range(episodes):
        #new training epoch each loop
        fourRoomsObj.newEpoch()

        #initial state
        currentState = fourRoomsObj.getPosition()
        packagesLeft = fourRoomsObj.getPackagesRemaining()

        #index for the next package to collect
        nextIndex = 0

        #make sure simulation still running
        while(fourRoomsObj.isTerminal() == False):
            #increment moves
            numMoves = numMoves + 1

            #state before action
            oldState = fourRoomsObj.getPosition()
            oldPackagesLeft = packagesLeft

            #implement epsilon decay
            currentEpsi = max(0.05, epsilon * (1 - episode / episodes))

            #choose next action - epsilon-greedy
            if random.random() < currentEpsi:
                #explore --> choose a random action
               action = random.randint(0, 3)
            else:
                #exploit --> choose the best action
                action = np.argmax(Q_values[convert(oldState, oldPackagesLeft), :])

            #take the action
            cellType, currentState, packagesLeft, isTerm = fourRoomsObj.takeAction(actionList[action])

            #reward functions:
            #if package collected - decrease in packages left
            if(packagesLeft < oldPackagesLeft):
                #check if correct package in sequence
                if (cellType == requiredOrder[nextIndex]):
                    reward = 200
                    nextIndex = nextIndex + 1
                #wrong package collected
                else:
                    reward = -500
                    #terminate epoch
                    fourRoomsObj._FourRooms__is_terminal = True
                    isTerm = True
            #out of bounds area
            elif(oldState == currentState):
                reward = -100
            #not a package or movement wasn't useful
            else:
                reward = -1

            totalReward = totalReward + reward

            #update Q values using the Q-learning update rule
            Q_values[convert(oldState, oldPackagesLeft), action] = Q_values[convert(oldState, oldPackagesLeft), action] + learning_rate * (
                reward + discount_factor * np.max(Q_values[convert(currentState, packagesLeft), :]) - 
                Q_values[convert(oldState, oldPackagesLeft), action]
            )

        #if completely done - all packages collected
        if( fourRoomsObj.isTerminal() and numMoves < bestSteps and nextIndex == 3):
            bestSteps = numMoves
            bestEpisode = episode

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