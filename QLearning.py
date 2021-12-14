import numpy as np

class QLearning:
    n = 0
    ThinIce = []
    location_to_state = {
        'L11': 0,
        'L12': 1,
        'L13': 2,
        'L14': 3,
        'L21': 4,
        'L22': 5,
        'L23': 6,
        'L24': 7,
        'L31': 8,
        'L32': 9,
        'L33': 10,
        'L34': 11,
        'L41': 12,
        'L42': 13,
        'L43': 14,
        'L44': 15,
    }

    rewards = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

    state_to_location = dict((state, location) for location, state in location_to_state.items())

    gamma = 0.75  # Discount factor
    alpha = 0.9  # Learning rate


    def __init__(self, learningRate, discoutFactor, filename):
        self.CreateTable(filename)
        self.learningRate = learningRate
        self.discoutFactor = discoutFactor

    def CreateTable(self, filename):
        f = open(filename)
        n = int(f.readline().replace("\n",""))
        self.n = n

        line = f.readline().replace("\n","")
        self.start_location = line

        line = f.readline().replace("\n", "")
        self.end_location = line

        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            self.ThinIce.append(line)

    def get_optimal_route(self,start_location, end_location):
        Q = np.zeros((self.n * self.n, self.n * self.n))

        rewards_new = np.copy(self.rewards)

        ending_state = self.location_to_state[end_location]

        rewards_new[ending_state, ending_state] = 999


        for i in range(1000):
            current_state = np.random.randint(0, 16)
            playable_actions = []
            for j in range(16):
                if rewards_new[current_state, j] > 0:
                    playable_actions.append(j)
            next_state = np.random.choice(playable_actions)
            TD = rewards_new[current_state, next_state] + self.gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[
                current_state, next_state]
            Q[current_state, next_state] += self.alpha * TD

        route = [start_location]

        next_location = start_location

        while (next_location != end_location):
            starting_state = self.location_to_state[start_location]
            next_state = np.argmax(Q[starting_state,])
            next_location = self.state_to_location[next_state]
            route.append(next_location)
            start_location = next_location
            if next_location in self.ThinIce:
                next_location = end_location
                route.append("DEAD")

        return route

    def compute(self):
        return self.get_optimal_route(self.start_location,self.end_location)
