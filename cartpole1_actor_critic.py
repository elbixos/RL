import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

gamma=0.98
max_episode=5000
env=gym.make("CartPole-v1")

prefix_log_file="log_actor_critic_dsum_"
id_file=0
while os.path.exists(prefix_log_file+str(id_file)+".csv"):
    id_file+=1
fichier_log=open(prefix_log_file+str(id_file)+".csv", "w")
print("Création du fichier de log", prefix_log_file+str(id_file)+".csv")

nbr_actions=2
nbr_inputs=4

def calcul_discount_rate(rewards_history, gamma, normalize=False):
    result=[]
    discounted_sum=0
    for r in rewards_history[::-1]:
        discounted_sum=r+gamma*discounted_sum
        result.insert(0, discounted_sum)

    # Normalisation
    if normalize is True:
        result=np.array(result)
        result=(result-np.mean(result))/(np.std(result)+1E-7)
        result=list(result)

    return result

def my_model(nbr_inputs, nbr_hidden, nbr_actions):
    entree=layers.Input(shape=(nbr_inputs,), dtype='float32')
    common=layers.Dense(nbr_hidden, activation="relu")(entree)
    common=layers.Dense(nbr_hidden, activation="relu")(common)
    action=layers.Dense(nbr_actions, activation="softmax")(common)
    critic=layers.Dense(1)(common)

    model=keras.Model(inputs=entree, outputs=[action, critic])
    return model


model=my_model(nbr_inputs, 32, nbr_actions)

optimizer=keras.optimizers.Adam(learning_rate=5e-4)
huber_loss=keras.losses.Huber()

m_reward=0
episode=0
score_history =[]
best_score = 0

for episode in range(max_episode):
    action_probs_history=[]
    critic_value_history=[]
    rewards_history=[]

    #print(optimizer.learning_rate)

    state=env.reset()[0]
    episode_reward=0
    with tf.GradientTape() as tape:

        # Récupération de données
        while True:
            action_probs, critic_value=model(np.expand_dims(state, axis=0))
            critic_value_history.append(critic_value[0, 0])
            action=np.random.choice(nbr_actions, p=np.squeeze(action_probs))
            #print(action)
            action_probs_history.append(action_probs[0, action])
            state, reward, done, truncated, infos=env.step(action)
            done = tf.logical_or(done,truncated)
            rewards_history.append(reward)
            episode_reward+=reward
            if done:
                break

        discount_rate=calcul_discount_rate(rewards_history, gamma, normalize=True)

        history=zip(action_probs_history, critic_value_history, discount_rate)
        actor_losses=[]
        critic_losses=[]
        total_losses=[]
        for action_prob, critic_value, discount_rate in history:
            actor_loss = -tf.math.log(action_prob)*(discount_rate-critic_value)
            critic_loss = huber_loss(np.expand_dims(critic_value, axis=0), np.expand_dims(discount_rate, axis=0))
            total_loss = tf.squeeze(actor_loss + critic_loss)
            #print(actor_loss, critic_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            total_losses.append(total_loss)

        #print (actor_losses, critic_losses)
        loss_value=tf.reduce_mean(total_losses)
        #print(loss_value)
        grads=tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode+=1
    m_reward=0.05*episode_reward+(1-0.05)*m_reward
    score_history.append(episode_reward)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
        model.save("model_actor_critic.keras")
        print("saved...",best_score)


    message="Episode {:04d}  score:{:6.1f}  MPE: {:6.1f} Average score :{:6.1f} Best score :{6.1f}"
    print(message.format(episode, episode_reward, m_reward, avg_score,best_score))
    #print(optimizer.learning_rate)


    fichier_log.write("{:f}:{:f}\n".format(episode_reward, m_reward))

    if m_reward>env._max_episode_steps-10:
        print("Fin de l'apprentissage".format(episode))
        break

fichier_log.close()
#model.save("model_actor_critic.keras")
