import pandas as pd
import matplotlib.pyplot as plt


one_DQN = pd.read_csv("results/DQN_for_MARL_15x15_more_one_agent_explore_2.csv")

# print(one_DQN.head(3))


# Assuming you have the DataFrame one_DQN containing the data
# Extracting relevant columns from the DataFrame
train_mean_steps = one_DQN['Train Steps']
test_mean_steps = one_DQN['Test Steps']
train_steps_std = one_DQN['Train Steps STD']
test_steps_std = one_DQN['Test Steps STD']

train_mean = one_DQN['Mean_Train']
test_mean = one_DQN['Mean_Test']
train_mean_std = one_DQN['STD_Train']
test_mean_std = one_DQN['STD_Test']



# Plotting
plt.figure(figsize=(10, 6))

# Plotting mean training steps with error bars representing standard deviation
plt.errorbar(range(len(train_mean_steps)), train_mean_steps, yerr=train_steps_std, label='Mean Train Steps', fmt='-o', color='blue')

# Plotting mean testing steps with error bars representing standard deviation
plt.errorbar(range(len(test_mean_steps)), test_mean_steps, yerr=test_steps_std, label='Mean Test Steps', fmt='-o', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Steps')
# plt.title('Mean Training and Testing Steps with Standard Deviation')
plt.legend()
plt.grid(True)
plt.savefig('results/mean_steps_one_agent.png')
plt.show()


# Plotting
plt.figure(figsize=(10, 6))

# Plotting mean training steps with error bars representing standard deviation
plt.errorbar(range(len(train_mean)), train_mean, yerr=train_mean_std, label='Mean Train', fmt='-o', color='blue')

# Plotting mean testing steps with error bars representing standard deviation
plt.errorbar(range(len(test_mean)), test_mean, yerr=test_mean_std, label='Mean Test', fmt='-o', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Steps')
# plt.title('Mean Training and Testing Reward with Standard Deviation')
plt.legend()
plt.grid(True)
plt.savefig('results/mean_reward_one_agent.png')
plt.show()


IQN_DF = pd.read_csv("results/DQN_for_MARL_15x15_more_four_agent_explore_2.csv")

# print(one_DQN.head(3))

# import matplotlib.pyplot as plt

# Assuming you have the DataFrame one_DQN containing the data
# Extracting relevant columns from the DataFrame
train_mean_steps = IQN_DF['Train Steps']
test_mean_steps = IQN_DF['Test Steps']
train_steps_std = IQN_DF['Train Steps STD']
test_steps_std = IQN_DF['Test Steps STD']

train_mean = IQN_DF['Mean_Train']
test_mean = IQN_DF['Mean_Test']
train_mean_std = IQN_DF['STD_Train']
test_mean_std = IQN_DF['STD_Test']



# Plotting
plt.figure(figsize=(10, 6))

# Plotting mean training steps with error bars representing standard deviation
plt.errorbar(range(len(train_mean_steps)), train_mean_steps, yerr=train_steps_std, label='Mean Train Steps', fmt='-o', color='blue')

# Plotting mean testing steps with error bars representing standard deviation
plt.errorbar(range(len(test_mean_steps)), test_mean_steps, yerr=test_steps_std, label='Mean Test Steps', fmt='-o', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Steps')
# plt.title('Mean Training and Testing Steps with Standard Deviation')
plt.legend()
plt.grid(True)
plt.savefig('results/mean_steps_four_agent.png')
plt.show()


# Plotting
plt.figure(figsize=(10, 6))

# Plotting mean training steps with error bars representing standard deviation
plt.errorbar(range(len(train_mean)), train_mean, yerr=train_mean_std, label='Mean Train', fmt='-o', color='blue')

# Plotting mean testing steps with error bars representing standard deviation
plt.errorbar(range(len(test_mean)), test_mean, yerr=test_mean_std, label='Mean Test', fmt='-o', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Steps')
# plt.title('Mean Training and Testing Reward with Standard Deviation')
plt.legend()
plt.grid(True)
plt.savefig('results/mean_reward_four_agent.png')
plt.show()
