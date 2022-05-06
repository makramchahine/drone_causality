This directory contains checkpoints that were used on the drone and for training the checkpoints used on the drone.

The directory chair4_fine contains the checkpoints that were actually flown on the drone and are identical to the checkpoints found in rosetta_drone/rnn_control/src/models. They were fine-tuned on the sliced sequences containing only the chair and synthetic sequences containing the chair. They started from chair4_long_balanced checkpoints. 

The directory chair4_long_balanced contains the starting checkpoints used to train the chair4_fine models. They were trained on long sequences, sliced sequences, and synthetic sequences. The chair4_fine models started training with these weights.