# RL Maths

Une compilation de ce que j'ai compris des maths en RL.

## quelques définitions

pour poser le problème, on considère que notre **agent** évolue dans un **environnement**, par le biais d'**actions**.
A un instant donné, l'agent est considéré comme étant dans un certain **état**.


on définit :
- $a$ : une action possible
- $s$ : un état.


par exemple, dans un monde déterministe, une action $a$ appliquée à un état $s$
va conduire à l'état $s'$.

l'environnement attribue à l'agent une certaine **récompense** instantanée
lorsque l'agent, dans un état donnée choisit telle action qui le conduit dans tel autre état.

- $r$ : la récompense associée au passage d'un état à un autre par le biais d'une action. Il s'agit d'une récompense instantannée.

Il s'agit donc de choisir, à partir d'un état initial, une série d'actions permettant d'obtenir une récompense cumulée maximale.

- $s_0$ : un état initial

- $\tau$ : une trajectoire. une trajectoire est l'enregistrement de l'état initial, suivi de toutes les actions suivantes jusqu'à un état terminal. $\tau = \{s_0, a_1,a_2,... a_T\}$

- $V(s)$ : la valeur d'un état. Quelle récompense cumulée (ou similaire) peut attendre l'agent qui passe par un état $s$.

- $Q(s,a)$ : la qualité d'un couple $s,a$. Si l'agent est dans un état $s$, quelle récompense cumulée (ou similaire) attend il de l'action $a$.

- $\pi$ : une politique. une fonction qui, en fonction de l'état, définit l'action à suivre. Dans un monde déterministe, on la note parfois $\mu$ et c'est une fonction $\mu(s) \rightarrow a$. Dans un monde stochastique, c'est une distribution de probabilité, qui à un état associe la probabilité de choisir chacune des actions possibles.

## Techniques

Commencons par les techniques adaptées aux **actions discrètes**

## Monte carlo learning :

On cherche à estimer les valeurs $V(s)$. Ceci peut par exemple être fait à l'aide d'un réseau de neurones. Le même raisonnement s'applique si l'on souhaite estimer $Q(s,a)$.

On initialise les $V(s)$ aléatoirement.
On définit une trajectoire, et on compte la récompense cumulée sur la trajectoire,
pour chaque instant de la trajectoire. Les récompenses antérieurs étant sans intérêt pour la valeur de V(s_t), cette récompense cumulée prend la forme suivante :

$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} ... + \gamma^{T-t-1} r_{T}$

On met à jour les valeurs des états $s_t$ avec la formule suivante, ou $\alpha$ est un taux d'apprentissage.
$V(s_t) = V(s_t) + \alpha (G_t - V(s_t))$

### Différence temporelle

Ici, on ne va pas forcément dérouler toute la trajectoire, mais une simple itération
va fournir les informations permettant la mise à jour des estimations (des valeurs ou des qualité).

Pour cela, on va auto estimer la fin de $G_t$. On le comprend en écrivant 
$G_t = r + \gamma G_{t+1}$


Si on suppose que V(s) fournit une estimation à peu près correcte de $G$,
on peut écrire la **différence temporelle**
$G_t = r_t + \gamma V(s_{t+1})$

Ceci fournira la base de SARSA et Qlearning

on peut imaginer une différence temporelle à plus d'une itération (disons 3 ou 10...)

### SARSA

C'est un exemple de stratégie **on policy**. Sarsa est le premier algo comme cela.

Disons  qu'on dispose d'une stratégie **epsilon-greedy**.

Quand on doit évaluer une qualité : on choisit une action, à l'aide de la policy actuelle. On doit alors estimer la valeur de l'état suivant, que l'on considère être la qualité de l'action suivante selon notre policy actuelle.

Dans un état $s$, on choisit l'action $a$, qui nous donne une récompense $r$ et nous amène dans l'état $s'$, pour lequel on choisit l'action $a'$.
(Sarsa vient de l'enchainement, à chaque itération de $s,a,r,s',a'$ )

$Q(s,a) = Q(s,a) + \alpha (r + \gamma Q(s',a') - Q(s,a))$

### Qlearning
C'est une stratégie off policy, au sens ou pour évaluer la valeur de l'état suivant,
on ne va pas re-appliquer notre politique, mais utiliser $max_{a'} Q(s',a')$.
Visiblement, ça change tout...

$Q(s,a) = Q(s,a) + \alpha (r + \gamma max_{a'} Q(s',a') - Q(s,a))$

### DQN
On utilise simplement un réseau de neurones pour evaluer $q(s,a)$.
Le réseau est décrit comme suit :
- en entrée : $s$
- en sortie : autant de neurones que d'actions possibles $a_i$. Chaque sortie
évalue $q(s,a_i)$

Avantage : Une modification des poids du réseau pour un état modifie le calcul
des sorties pour les autres états. On explore plus vite. Chaque état "diffuse" sur les autres. Cela permet d'estimer des espaces de grande dimension.

Si on écrit le $y$ dy réseau (au sens apprentissage supervisé), c'est :
$y = r + \gamma max_{a'} Q(s',a')$

Si on écrit le $y_pred$ dy réseau (au sens apprentissage supervisé), c'est :
$y = Q(s,a)$

l'erreur du réseau est bien, comme précédemment :
$error = r + \gamma max_{a'} Q(s',a') - Q(s,a)$

Inconvénient : une modification du réseau modifie $Q(s,a)$, mais aussi
$Q(s,a')$ => risque d'instabilité. => on utilise deux réseaux jumeaux

- réseau online : pour calculer $Q(s,a)$
- réseau target : pour calculer $Q(s,a')$

de temps en temps (tous les $C$ steps), on copie online -> target.

### DDQN

### Stochastic reward
Si les récompenses sont probabilistes, on parle de **distributional Q learning**.
(C51 est un exemple)

### Rainbow DQN
Chacune de ces variantes ajoutées.

## Actions continues

