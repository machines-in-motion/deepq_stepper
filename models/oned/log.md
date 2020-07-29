## This file contains information regarding the models stored

1) dqs-1 : This model is trained with cost abs(x-u) + abs(xd) + 100 (if terminates). The step time for this model is 0.1 sec, number of actions = 9.
	ANN - 3 layers 1024 each.
2) dqs-2 : This model is trained with cost abs(x-u) + abs(xd) + abs(u) + 100(if terminates). The step time for this model is 0.1 sec, number of actions = 9
	ANN - 3 layers 1024 each
3) dqs-3 : This model is trained with the same cost as above. The step time is 0.1 sec, number of actions 21. 
	ANN - 4 layers 2048 each
