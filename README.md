# AlphaZero

This project is an implementation of the AlphaZero algorithm for the game Connect N.

### Required packages:
- numpy
- python
- pytorch



### Starting the program:
- hyperparameters and settings for the algorithm can be changed in the file "Hyperparameter.py".
- use the file "Main.py":
    - change the variables show_output to true for having a gui
    - choose one of the 3 options:
      1. Play a tournament between different versions of the algorithm including alphaZero. MCTS and random
      2. PLay against the computer (choose your oponent amongst mcts, random, user and alphazero, which can only be chosen if a trained version exists)
      3. Train an AlphaZero agent, which creates and saves new versions of alphazero that can be used afterwards
- run the file "Main.py"

