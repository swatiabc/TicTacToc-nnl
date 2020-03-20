# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:34:04 2020

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

import numpy as np 
import pandas as pd 
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import pprint
import random

from keras.models import load_model
model=load_model("TicTacToe.h5")

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
    #pprint.pprint(scores)
    #print("max",next_coordinate)
    #pprint.pprint(legal_moves)
    score=scores[next_coordinate]
    return next_coordinate,next_board_state,score

'''
print("Self learning tictactoe using neural networks")
print("enter 1 to play and 0 to exit")
choice=int(input())

while(choice==1):
    print("starting game")
    game=TicTacToe()
    game.toss()
    print(game.board)
    print(game.turn_monitor,"will play first")
    while(1):
        if game.game_status()==4 and game.turn_monitor==0:
            print("your turn")
            while(1):
                try:
                    print("enter row,column number to put 0")
                    coord=input()
                    coordinates = eval(coord)
                    print(coordinates)
                    game_status,board=game.move(0,coordinates)
                    print(board)
                    break;
                except:
                    print("invalid move")
        elif game.game_status()==4 and game.turn_monitor==1:
            print("program's turn")
            move,board_state,score=move_selector(model,game.board,game.turn_monitor)
            game_status,board=game.move(game.turn_monitor,move)
            print(board)
            #print(score)
            #print(game.game_status())
        else:
            #print(game.game_status())
            break
    if game.game_status()==2 and (1-game.turn_monitor)==1:
        print("program won")
    if game.game_status()==2 and (1-game.turn_monitor)==0:
        print("opponent won")
    if game.game_status()==3:
        print("draw")
    print("to play again enter 1")
    choice=int(input())

'''
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
            #3opponent=opponent_move_selector(game.board, game.turn_monitor, mode)
            print("your turn")
            while(1):
                try:
                    print("enter row,column number to put 0")
                    coord=input()
                    coordinates = eval(coord)
                    print(coordinates)
                    game_status,board=game.move(0,coordinates)
                    print(board)
                    break;
                except:
                    print("invalid")
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

print("Self Learning TicTac Toe using NN")
print("enter 1 to start")
choice=int(input())
while(choice==1):
    model,y,result=train(model, mode=5,print_progress=True)
    print("to play again enter 1")
    choice=int(input())