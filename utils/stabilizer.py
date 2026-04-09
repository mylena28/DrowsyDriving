from collections import deque

class StateStabilizer:
    def __init__(self, window_size=15, change_threshold=5):
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)

        # média móvel
        stable_value = sum(self.window) / len(self.window)

        return stable_value
