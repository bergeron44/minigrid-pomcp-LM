import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-DoorKey-6x6-v0")
print("Wrapped Env Type:", type(env))
print("Unwrapped Env Type:", type(env.unwrapped))
print("Wrapped Attributes:\n", dir(env))
print("\nUnwrapped Attributes:\n", dir(env.unwrapped))
