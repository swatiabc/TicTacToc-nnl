# TicTacToc-nnl
Self learning Tic tac toe game using neural networks.

## TicTacToe_opponent.py
Our program is trained by playing with opponent 200,000 times. Most favorable board state is given higher weight and 
least favorable has less weight. After few games the program learns to give more favorable board state higher weight.

## TicTacToe_human.py
This is our program which uses model trained from playing with the opponent. The model predicts scores and next move for the program.

## TicTacToe.h5
This is model created when programs plays with opponent 200,000 times. This model is used by the program to predict it's next move while 
playing with human.

## TicTacToe.csv
This is the list of wins, loss, and draws of the program against opponent.

## Graphs
![graph for wins,losses,draws](images/TicTacToe_graph.png)
7 - losses<br>
3 - draws<br>
2 - wins<br>

As the number of games increases, number of losses decreases and number of wins and draws increases  

## Output
Program plays 1<br>
Opponent plays 0<br>
toss is random<br>
-1 is blank<br>

![output 1](images/Capture1.JPG)
![output 2](images/Capture2.JPG)
![output 3](images/Capture3.JPG)
![output 4](images/Capture4.JPG)

## References
https://dhanushkishore.github.io/a_self_learning_tic-tac-toe_player/ <br>
https://en.wikipedia.org/wiki/Tic-tac-toe
