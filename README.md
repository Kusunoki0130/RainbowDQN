# RainbowDQN

实现教程：https://github.com/Curt-Park/rainbow-is-all-you-need

用到的 site-packages

> torch           
>
>  matplotlib           
>
> pyglet               
>
> gym                  
>
> PyVirtualDisplay 
>
> moviepy 
>
> pygame 

选择的 environment："CartPole-v1"（后续有时间可以再跑其他的）

代码在 main 分支，实验结果在 result 分支

代码说明：utils 中包含了经验回放池和网络结构的实现，其余文件为 dqn 模型的具体实现

结果说明：

- dqn_i.pkl：模型参数，使用方法已经写在代码中

- scores_i.txt：训练中，每一个 episode 的得分

- losses_i.txt：每一个 batch 的损失

- epsilons_i.txt：ε-greedy 策略中 ε 的变化（某些改进算法没有该文件）

载入参数后，调用 test 可以生成演示视频
