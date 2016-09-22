# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:17:22 2016

@author: Huynh
"""

import numpy as np
import matplotlib.pylab as plt

plt.ion()

# Um ein Spiel zu starten:
#
#game = ConnectFour(1, -1),     1 = Player 1, -1 = Player 2
#
#Diese Funktion ist zum Spielen, in ne Loop bringen
#game.play(col, playernr),      col = Säule, wo der Disc platziert wird, playernr: 1 -> Player1, 2-> Player2 (NICHT -1)
#
#Wenn Spiel zu Ende ist:
#game.newGame()
#
#Um den State abzufragen:
#game.getGame()

class ConnectFour():
    
    def __init__(self, player1, player2, plot = False):
        self.game = dict()
        self.newGame()
        self.p = dict()
        self.p[1] = player1
        self.p[2] = player2
        self.turn = True
        self.plot = plot
        self.nTurns = 0
        if plot == True:
            self.window = plt.imshow(self.game[0], cmap='Greys_r', interpolation = 'none', origin='lower')

    def newGame(self, gameNr = 0):
        self.game[gameNr] = np.zeros((7,7))
        self.turn = True
        self.nTurns = 0

    def dropDisc(self, col, player, gameNr = 0):
        height = self.getLegitRow(col, gameNr)
        
        if height == -1:
            return -1
        
        self.placeDisc(col, height, player, gameNr)
        
        return 1
        
    
    def getGame(self, gameNr = 0):
        return self.game[gameNr]
        
    def getLegitRow(self, col, gameNr = 0):
        row = self.getGame(gameNr)[:,col]

        legitRow = 0        
        
        for element in row:
            if element != 0:
                legitRow += 1
            else:
                break

        
        if legitRow > 5:
            return -1
        
        return legitRow


    def placeDisc(self, col, row, player, gameNr = 0):
        self.game[gameNr][row][col] =  player     
        self.nTurns += 1
        
    def checkWin(self, player, gameNr = 0):
        filtered = [[cell == player for cell in row] for row in self.game[gameNr]]

        flag = self.checkDiagWin(filtered)
        
        if flag == True:
            return True            
            
        flag = self.checkVertWin(np.transpose(filtered))
        
        if flag == True:
            return True       
            
        flag = self.checkVertWin(filtered)
        
        if flag == True:
            return True       
            
        flag = self.checkDiagWin(np.fliplr(filtered))
        
        if flag == True:
            return True          
        return False
        
    def checkDiagWin(self, game):
        for i in range(3,6):
            col = i
            row = 0
            flag = 0
            for _ in range(i+1):
                if game[col][row] == True:
                    flag += 1
                col -= 1
                row += 1
                
            if flag > 3:
                return True
            
        return False
        
    def checkVertWin(self, game):
        for row in game:
            flag = 0
            for cell in row:
               if cell == True:
                   flag += 1
                   
            if flag > 3:
                return True
            
        return False        
                       
    def changeTurn(self):
        self.turn = not self.turn             

    def getTurn(self):
        if self.turn == True:
            return 1
        if self.turn == False:
            return 2
    
    def play(self, col, playerNr, gameNr = 0):
        # return -1 = Eingabe wurde nicht angenommen --> siehe Konsole
        # return 0 = Spiel geht ohne besonders Ereignis weiter
        # return -2 = Unentschieden
        # return 1/2 = Spieler 1/2 gewinnt
        if playerNr != self.getTurn():
            print 'Wrong player played.'
            return -1

        if col < 0:
            print 'Only use colloumns from 0 to 6.'
            return -1
        
        if col > 6:
            print 'Only use colloumns from 0 to 6.'
            return -1
        
        self.dropDisc(col, self.p[playerNr], gameNr)

        win = self.checkWin(self.p[playerNr])
        if win == True:
            #print 'Player {} won!'.format(playerNr)
            return playerNr                
                
        if self.nTurns == 42:
            return -2
        
            self.changeTurn()
            
        return 0
    
    def updateScreen(self, gameNr = 0):
        self.window.set_data(self.game[gameNr])
        plt.draw()
        plt.pause(0.1)

    def manualGameMode(self):
        end = False
        while end != True:
            print 'Choose your coloumn'
            inp = int(raw_input())
            turn = self.getTurn()
            end = self.play(inp, turn)
            
            
            if self.plot == True:
                self.updateScreen()
    
#game = ConnectFour(1, 2, plot = False)
#game.manualGameMode()


