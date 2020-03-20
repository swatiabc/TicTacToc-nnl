# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:24:14 2020

@author: HP
"""
"""
0=opponent/human
1=player
2=win
3=draw
4=in progress
5=hard
6=easy
7=loss
"""
"""
1. win
2. block
3. fork
4. block fork
5. center
6. opposite corner
7. empty corner
8. empty side
9. hard
10. 
"""

import numpy as np 
import pandas as pd 
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import pprint
import random

shapes=3

class TicTacToe(object):
    def __init__(self):
        self.board = np.full((3,3),-1)
    def toss(self):
        self.turn_monitor = random.choice([0,1])
        return self.turn_monitor
    def move(self,player,coordinate):
        if self.board[coordinate]!=-1 or self.game_status()!=4 or self.turn_monitor!=player:
            raise ValueError("wrong move")
        self.board[coordinate]=player
        self.turn_monitor=1-player
        return self.game_status(),self.board
    def game_status(self):
        ##row, column
        for i in range(shapes):
            if -1 not in self.board[i,:] and len(set(self.board[i,:]))==1:
                return 2
            if -1 not in self.board[:,i] and len(set(self.board[:,i]))==1:
                return 2
        ##diag
        if -1 not in np.diag(self.board) and len(set(np.diag(self.board)))==1:
            return 2
        if -1 not in np.diag(np.fliplr(self.board)) and len(set(np.diag(np.fliplr(self.board))))==1:
            return 2
        ##draw
        if -1 not in self.board:
            return 3
        return 4

def legal_moves_generator(current_board_state,turn_monitor):
    legal_moves={}
    for i in range(shapes):
        for j in range(shapes):
            if current_board_state[i,j]==-1:
                new_board_state = current_board_state.copy()
                new_board_state[i,j]=turn_monitor
                legal_moves[(i,j)]=new_board_state.flatten()
    return legal_moves

def move_selector(model,current_board_state,turn_monitor):
    scores={}
    legal_moves=legal_moves_generator(current_board_state, turn_monitor)
    for coordinates in legal_moves:
        score=model.predict(legal_moves[coordinates].reshape(1,9))
        scores[coordinates]=score
    next_coordinate=max(scores,key=scores.get)
    next_board_state=legal_moves[next_coordinate]
    score=scores[next_coordinate]
    return next_coordinate,next_board_state,score
    
"""opponent's move"""

class opponent_move(object):
    def __init__(self,current_board_state,legal_moves,turn_monitor):
        self.current_board_state=current_board_state
        self.legal_moves=legal_moves
        self.turn_monitor=turn_monitor
        
    def row_win_move(self,board_state_copy,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=board_state_copy.copy()
            board_state[coordinates] = turn_monitor_copy
            for i in range(shapes):
                if -1 not in board_state[i,:] and len(set(board_state[i,:]))==1:
                    return coordinates

    def column_win_move(self,board_state_copy,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=board_state_copy.copy()
            board_state[coordinates] = turn_monitor_copy
            for i in range(shapes):
                if -1 not in board_state[:,i] and len(set(board_state[:,i]))==1:
                    return coordinates

    def diagonal1_win_move(self,board_state_copy,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=board_state_copy.copy()
            board_state[coordinates] = turn_monitor_copy
            #print(coordinates,board_state)
            if -1 not in np.diag(board_state) and len(set(np.diag(board_state)))==1:
                return coordinates

    def diagonal2_win_move(self,board_state_copy,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=board_state_copy.copy()
            board_state[coordinates]=turn_monitor_copy
            if -1 not in np.diag(np.fliplr(board_state)) and len(set(np.diag(np.fliplr(board_state))))==1:
                return coordinates

    def row_block_move(self,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=self.current_board_state.copy()
            board_state[coordinates]=turn_monitor_copy
            i=coordinates[0]
            if -1 not in board_state[i,:] and (board_state[i,:]==1-turn_monitor_copy).sum()==2:
                return coordinates

    def column_block_move(self,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=self.current_board_state.copy()
            board_state[coordinates]=turn_monitor_copy
            i=coordinates[1]
            if -1 not in board_state[:,i] and (board_state[:,i]==1-turn_monitor_copy).sum()==2:
                return coordinates

    def diagonal1_block_move(self,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=self.current_board_state.copy()
            board_state[coordinates] = turn_monitor_copy
            if (coordinates[0]-coordinates[1])==0:
                if -1 not in np.diag(board_state) and (np.diag(board_state)==1-turn_monitor_copy).sum()==2:
                    #print(board_state)
                    return coordinates

    def diagonal2_block_move(self,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        for coordinates in self.legal_moves:
            board_state=self.current_board_state.copy()
            board_state[coordinates]=turn_monitor_copy
            if (coordinates[0]+coordinates[1])==2:
                if -1 not in np.diag(np.fliplr(board_state)) and (np.diag(np.fliplr(board_state))==1-turn_monitor_copy).sum()==2:
                    return coordinates

    def fork_win_move(self,check_fork=0):
        if check_fork==0:
            turn_monitor_copy=self.turn_monitor
        else:
            turn_monitor_copy=1-self.turn_monitor

        win_count=0
        fork_coordinates=[]
        win_moves=[self.row_win_move,self.column_win_move,self.diagonal1_win_move,self.diagonal2_win_move]
        for coordinates in self.legal_moves:
            win_count=0
            board_state=self.current_board_state.copy()
            board_state[coordinates]=turn_monitor_copy
            for fn in win_moves:
                if fn(board_state):
                    win_count=win_count+1
                    #print("potential fork coor:",coordinates)
            if win_count>1:
                #print("fork exits",coordinates," win:",win_count)
                fork_coordinates.append(coordinates)
        random.shuffle(fork_coordinates)
        #print(fork_coordinates)
        return fork_coordinates

    def center_corner_move(self):
        if self.current_board_state[1,1]==-1:
            return (1,1)
        if self.current_board_state[0,0]==1-self.turn_monitor and self.current_board_state[2,2]==-1:
            return (2,2)
        if self.current_board_state[2,2]==self.turn_monitor and self.current_board_state[0,0]==-1:
            return (0,0)
        if self.current_board_state[0,2]==1-self.turn_monitor and self.current_board_state[2,0]==-1:
            return (2,0)
        if self.current_board_state[2,0]==self.turn_monitor and self.current_board_state[0,2]==-1:
            return (0,2)        


def opponent_move_selector(current_board_state,turn_monitor,mode):
    legal_moves=legal_moves_generator(current_board_state, turn_monitor)
    opponent=opponent_move(current_board_state, legal_moves, turn_monitor)
    win_moves=[opponent.row_win_move,opponent.column_win_move,opponent.diagonal1_win_move,opponent.diagonal2_win_move]
    block_moves=[opponent.row_block_move,opponent.column_block_move,opponent.diagonal1_block_move,opponent.diagonal2_block_move]
    if mode==5:
        random.shuffle(win_moves)
        random.shuffle(block_moves)
        for fn in win_moves:
            if fn(opponent.current_board_state):
                #print("win:")
                return fn(opponent.current_board_state)
        for fn in block_moves:
            if fn():
                #print("block:",fn)
                return fn()
        if opponent.fork_win_move():
            #print("fork")
            fork=opponent.fork_win_move()
            return fork[0]
        if opponent.center_corner_move():
            #print("ccm")
            return opponent.center_corner_move();

        selected_move=random.choice(list(legal_moves.keys()))
        #print("random")
        return selected_move
    elif mode==6:
        selected_move=random.choice(list(legal_moves.keys()))
        return selected_move


def train(model,mode,print_progress=False):
    game=TicTacToe()
    a=game.toss()
    #print(a)
    scores=[]
    updated_scores=[]
    board_states=[]
    result=-1
    game_status=-1
    while(1):
        if game.game_status()==4 and game.turn_monitor==1:
            selected_move,new_board_state,score=move_selector(model,game.board,game.turn_monitor)
            scores.append(score[0][0])
            board_states.append(new_board_state)
            game_status,board=game.move(game.turn_monitor,selected_move)
            if print_progress==True:
                print("programs's move")
                print(board)
                print("\n")
        elif game.game_status()==4 and game.turn_monitor==0:
            opponent=opponent_move_selector(game.board, game.turn_monitor, mode)
            game_status,board=game.move(game.turn_monitor,opponent)
            if print_progress==True:
                print("opponent's move")
                print(board)
                print("\n")
        else:
            break
    board_states=tuple(board_states)
    board_states=np.vstack(board_states)
    if game_status==2 and game.turn_monitor==0:
        updated_scores=shift(scores,-1,cval=1.0)
        result=2
    elif game_status==2 and game.turn_monitor==1:
        updated_scores=shift(scores,-1,cval=-1.0)
        result=7
    elif game_status==3:
        updated_scores=shift(scores,-1,cval=0)
        result=3
    if print_progress==True:
        print("program has",result)
        print("correcting score and update weights")
    x=board_states
    y=updated_scores
    
    def shuffle_array(a,b):
        a=np.array(a)
        b=np.array(b)
        assert len(a)==len(b)
        p=np.random.permutation(len(a))
        return a[p],b[p]
    
    x,y=shuffle_array(x,y)
    x=x.reshape(-1,9)
    model.fit(x,y,epochs=1,batch_size=1,verbose=0)
    return model,y,result


from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(18,input_dim=9,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(9,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,kernel_initializer='normal'))

learning_rate=0.001
momentum=0.8
sgd = SGD(lr=learning_rate,momentum=momentum,nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)
#model.summary()

"""testing"""
'''
game = TicTacToe()
game.toss()
# choose the first move
print("Player assigned mark 1",game.turn_monitor," won the toss")
print("Initial board state:")
print(game.board)
selected_move,new_board_state,score=move_selector(model,game.board,game.turn_monitor)
print("Selected move: ",selected_move)
print("Resulting new board state: ",new_board_state)
print("Score assigned to above board state by Evaluator(model): ", score)
selected_move[0]
#s=[1,2,3,4]
'''

#updated_model,y,result=train(model, mode=5,print_progress=True)

data_for_graph=pd.DataFrame()

for count in range(200000):
    mode=np.random.choice([6,5],p=[0.4,0.6])
    model,y,result=train(model, 5,print_progress=True)
    data_for_graph=data_for_graph.append({"game_counter":count,"result":result},ignore_index=True)
    if count % 10000 == 0:
        print("game#:",count)
        print(mode)
    count+=1
    
bins = np.arange(1, count/10000) * 10000
data_for_graph['game_counter_bins'] = np.digitize(data_for_graph["game_counter"], bins, right=True)
counts = data_for_graph.groupby(['game_counter_bins', 'result']).game_counter.count().unstack()
ax=counts.plot(kind='bar', stacked=True,figsize=(17,5))
ax.set_xlabel("Count of Games in Bins of 10,000s")
ax.set_ylabel("Counts of Draws/Losses/Wins")

data_for_graph.to_csv("TicTacToe.csv",index=False)

model.save("TicTacToe.h5")