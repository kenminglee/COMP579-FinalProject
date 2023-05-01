class BaseAgent:
    def choose_action(self, o, train=True):
        raise NotImplementedError

    def learn(self, s, a, r, s_, terminated, truncated, info):
        raise NotImplementedError