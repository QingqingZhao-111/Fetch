from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
env_name = "CartPole-v0"
env = gym.make(env_name)
# 导入模型
model = DQN.load("./model/CartPole.pkl")

state = env.reset()
done = False
score = 0
while not done:
    # 预测动作
    action, _ = model.predict(observation=state)
    # 与环境互动
    state, reward, done, info = env.step(action=action)
    score += reward
    env.render()
env.close()
print("score=",score)
