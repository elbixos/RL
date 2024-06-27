# RL Repository

some tests about **Reinforcement Learning**.

I, for now, just used the *Gymnasium package* to get environnements.

## Cartpole Environment

My firt attempts were with *cartpole v1* which is continuous.
Many of the code that I found was using cartpole v0 and it needs some adjustments
to work.

for cartpole, I have, working :

- *cartpole1_Qlearning_tabular*. demonstrate the use of Qlearning in a tabular version. It was my first attempt. It is based on some code found somewhere (shame on me) but slightly modified to make it work.

- *cartpole1_DQN*. demonstrate the use of a DQN version of Qlearning. It is also based on some code found somewhere (shame on me) but really heavily modified to make it work. The DQN has 2 models (1 target), and uses memory and replay.

- *cartpole1_actor_critic* will be my next target, and based on [L4Project video on the subject](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmJ6NVR3NExuUkNRV21VNUlISXdxbGxxM2dPUXxBQ3Jtc0ttU2oyVTEwMU0yYmxvTWNRRTBjbHMtOGFSWjhIS3QtMWVZd0dyb0hPVTR4WmVrcUtVYmh2LWJaRVZSTlp2ZVI4dmlFdTVFeXRjQkExd2twTnpsSHlfUW50YVJndDFQRW5ubVdpSTVIOG54SkU1X0FmTQ&q=https%3A%2F%2Fgithub.com%2FL42Project%2FTutoriels%2Ftree%2Fmaster%2FDivers%2Frenforcement6&v=1okjkEMP79c)

Globally, L4Project is a very good source of information in french, for reinforcement learning