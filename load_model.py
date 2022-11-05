import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO


# loading the model
model = PPO.load('./train/best_model_50000.zip')

# creating the mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# Simplyfying the control
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Grayscale
env = GrayScaleObservation(env, keep_dim=True)

# wrap inside the dummy environment
env = DummyVecEnv([lambda: env])

# Stack the frames
env = VecFrameStack(env, 4, channels_order='last')


state = env.reset()

while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()