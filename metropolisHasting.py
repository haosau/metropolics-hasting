# 开发时间 2023/12/22 17:25
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

# 目标概率密度函数（Beta分布）
def target_distribution_beta(x, alpha=2, beta_parm=5):
    return beta.pdf(x, alpha, beta_parm)

# 提议分布的采样函数（正态分布）
def proposal_sampler(x, sigma=1.0):
    return np.random.normal(loc=x, scale=sigma)

# Metropolis-Hastings算法
def metropolis_hastings(num_samples, initial_sample,proposal_sigma):
    samples = [initial_sample]
    for _ in range(num_samples - 1):
        current_sample = samples[-1]
        proposed_sample = proposal_sampler(current_sample, sigma=proposal_sigma)
        A= target_distribution_beta(proposed_sample)*proposal_sampler(proposed_sample, sigma=proposal_sigma)\
           /target_distribution_beta(current_sample)/proposal_sampler(current_sample, sigma=proposal_sigma)
        acceptance_ratio = min(1,A)
        if np.random.uniform(0, 1) < acceptance_ratio:
            samples.append(proposed_sample)
        else:
            samples.append(current_sample)
    return np.array(samples)

# 绘制目标概率密度函数的图形
plt.subplot(1,2,1)
x_values = np.linspace(-5, 5, 1000)
target_values = target_distribution_beta(x_values)
plt.plot(x_values, target_values, label='Target Distribution')
plt.xlabel("sample")
plt.ylabel("probably density")
plt.xlim([-2 ,4])
plt.ylim([0,3])
plt.title("Target Distribution")
# 运行Metropolis-Hastings算法并绘制样本直方图
plt.subplot(1,2,2)
num_samples = 10000
initial_sample = 0.0
proposal_sigma = 1.0
plt.xlabel("sample")
plt.ylabel("probably density")
plt.xlim([-2 ,4])
plt.ylim([0,3])
samples = metropolis_hastings(num_samples, initial_sample, proposal_sigma)
plt.hist(samples, bins=500, density=True, alpha=0.5, label='Sample Histogram')

plt.legend()
plt.title('Metropolis-Hastings Sampling')
plt.show()
