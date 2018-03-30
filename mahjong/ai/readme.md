
# AI Mahjong player. 

This is a mahjong ai for studying and experiment.We use monte tree to write a supervised algorithm of playing mahjong.The algo use monte tree to simulate game several times after which it make decision of how to playing a tile by selecting the most selected decision.Then a CNN net will be trained,and the trained model will be used to make decision and playing with the previous algo until the CNN model get stronger enough eg.to have a win rate of more than 60%.Then the previous algo will be replaced by the trianed one,And repeate the training again and again.

## Margin attack strategy using monte tree ##
In the first stage,the task is to write a fixed strategy to make decision using monte tree.In the code,i use a state machine to control the process of simulation of playing mahjong in monte tree.The simulation will randomly run SIM_NUM times,And update the tree with the score of each simulation's result.  

## CNN model to predicate the best tile ##
The second version is to use CNN model to predicate the probability with which the ai will gain the ability to learn by self.The ai will play with self continualy and update the CNN model.Once the continually updated model earns a win rate of more than 60%,we'll save the model as a new standard model.We may use a gpu cluster to train the model.

## contribution is welcome ##
Contacting me through message or email:yuanp0813@163.com