import numpy as np
import pandas as pd
import os
import itertools
from typing import Tuple

def score_deck(deck: str,
               seq1: str,
               seq2: str) -> Tuple[int, int, int, int]:
    '''
    Given a shuffled deck of cards, a sequence chosen by player two (me), and a sequence chosen by player two (opponent), 
    return the number of cards/tricks for each variation of Penney's Game.
    
    Arguments:
        - deck (str): randomly shuffled deck of 52 cards
        - seq1 (str): the 3-card sequence chosen by player 1 (opponent) (ex. BBB, RBR)
        - seq2 (str): the 3-card sequence chosen by player 2 (me) (ex. RRR, BRB)

    Outputs:
        - p1_cards (int): the number of cards player 1 (opponent) won
        - p2_cards (int): the number of cards player 2 (me) won
        - p1_tricks (int): the number of tricks player 1 (opponent) won
        - p2_tricks (int): the number of tricks player 2 (me) won
    '''
    p1_cards = 0
    p2_cards = 0
    pile = 2 # because we are starting at the third position
    
    p1_tricks = 0
    p2_tricks = 0
    
    i = 0
    while i < len(deck) - 2: # iterate through the deck, adding 1 to the pile for each step, then check the current sequence
        pile += 1
        current_sequence = deck[i:i+3]
      # if the sequence matches either player's sequence, add the current pile to their cards and restart the pile. 
      # add one to their tricks and move forward three cards in the deck.
        if current_sequence == seq1:
            p1_cards += pile
            pile = 2
            p1_tricks += 1
            i += 3
        elif current_sequence == seq2:
            p2_cards += pile
            pile = 2 
            p2_tricks += 1
            i += 3
        else: # if no one wins just move through the deck
            i += 1

    return p1_cards, p2_cards, p1_tricks, p2_tricks


def calculate_winner(p1_cards: int,
                     p2_cards: int,
                     p1_tricks: int,
                     p2_tricks: int)-> Tuple[int, int, int, int]:
        '''
        Given the number of cards and tricks for each player, calculate who wins for cards and tricks, as well as draws for cards and tricks.
        If player one (opponent) wins, the winner is set to 0. If player 2 (me) wins, the winner is set to 1.
        Also indicates if there was a draw.

        Arguments:
            - p1_cards (int): number of cards player 1 won
            - p2_cards (int): number of cards player 2 won
            - p1_tricks (int): number of tricks player 1 won
            - p2_tricks (int): number of tricks player 2 won
        
        Output:
            - cards_winner (int): specifies who won based on cards
            - cards_draw (int): 1 if a draw occurred, 0 otherwise
            - tricks_winner (int): specifies who won based on tricks
            - tricks_draw (int): 1 if a draw occured, 0 otherwise
        '''
        # initializations only change if p2 wins or there is a draw
        cards_winner = 0
        cards_draw = 0
        tricks_winner = 0
        tricks_draw = 0

        # if p2 wins, set winner to 1, otherwise it is 0 (including draws)
        # if there is a draw, set draw counter to 1
        if p1_cards < p2_cards:
            cards_winner = 1
        elif p1_cards == p2_cards:
            cards_draw = 1
        if p1_tricks < p2_tricks:
            tricks_winner = 1
        elif p1_tricks == p2_tricks: 
            tricks_draw = 1
        return cards_winner, cards_draw, tricks_winner, tricks_draw


def play_one_deck(deck: str,
                  data: str,
                  batched: bool = False):
    '''
    For a single deck, this function plays every possible combination of sequences for both players and saves the outcome as .npy arrays.
    
    Arguments:
        - deck (str): a string of either 0 or 1 representing a generated deck.
        - data (str): the data file information is being saved to.
        - batched (bool): whether to save each file individually or not, in the case of batch saving for efficiency

    Output:
        - p2_wins_cards, p2_wins_tricks (np.ndarray): arrays of card and trick wins for p2
        - draws_cards, draws_tricks (np.ndarray): arrays of card and trick ties
    '''
    sequences = ['000', '001', '010', '011', '100', '101', '110', '111'] # possible selections for either player
    combinations = itertools.product(sequences, repeat=2) # create all possible combinations of p1 and p2's choices
    p2_wins_cards = pd.DataFrame(columns=sequences, index=sequences)
    p2_wins_tricks = pd.DataFrame(columns=sequences, index=sequences)
    draws_cards = pd.DataFrame(columns=sequences, index=sequences)
    draws_tricks = pd.DataFrame(columns=sequences, index=sequences)

    for seq1, seq2 in combinations: # for each possible play sequence for p1 and p2: score the deck versus the selections, calculate the winner, and then insert the wins 
        p1_cards, p2_cards, p1_tricks, p2_tricks = score_deck(deck, seq1, seq2)
        cards_winner, cards_draw, tricks_winner, tricks_draw = calculate_winner(p1_cards, p2_cards, p1_tricks, p2_tricks)
        # insert wins at the location on the DataFrame corresponding to the two sequences
        p2_wins_cards.at[seq1, seq2] = cards_winner
        draws_cards.at[seq1, seq2] = cards_draw
        p2_wins_tricks.at[seq1, seq2] = tricks_winner
        draws_tricks.at[seq1, seq2] = tricks_draw
    
    deck_name = str(int(deck, 2)) # convert the binary deck to a number to name the results files

    # Reorder rows to start with '000' in the bottom-left corner
    reversed_sequences = sequences[::-1]
    p2_wins_cards = p2_wins_cards.loc[reversed_sequences, :]
    p2_wins_tricks = p2_wins_tricks.loc[reversed_sequences, :]
    draws_cards = draws_cards.loc[reversed_sequences, :]
    draws_tricks = draws_tricks.loc[reversed_sequences, :]

    if batched:
        return p2_wins_cards, p2_wins_tricks, draws_cards, draws_tricks

    else:
        # save the results to different folders for each variation
        np.save(f'{data}/cards/{deck_name}.npy', p2_wins_cards, allow_pickle = True)
        np.save(f'{data}/tricks/{deck_name}.npy', p2_wins_tricks, allow_pickle = True)
        np.save(f'{data}/cards_ties/{deck_name}.npy', draws_cards, allow_pickle = True)
        np.save(f'{data}/tricks_ties/{deck_name}.npy', draws_tricks, allow_pickle = True)
    return p2_wins_cards, p2_wins_tricks, draws_cards, draws_tricks

def batch_save(batch_cards: np.ndarray,
               batch_tricks: np.ndarray,
               batch_cards_ties: np.ndarray,
               batch_tricks_ties: np.ndarray,
               name: str,
               data: str = 'data/'):
    '''
    Saves cards and tricks for batched data.

    Arguments:
        - batch_cards, batch_tricks, batch_cards_ties, batch_tricks_ties (np.ndarray): card wins, trick wins, card ties, and trick ties for the batch
    '''
    np.save(f'{data}/cards/batch_{name}.npy', batch_cards, allow_pickle = True)
    np.save(f'{data}/tricks/batch_{name}.npy', batch_tricks, allow_pickle = True)
    np.save(f'{data}/cards_ties/batch_{name}.npy', batch_cards_ties, allow_pickle = True)
    np.save(f'{data}/tricks_ties/batch_{name}.npy', batch_tricks_ties, allow_pickle = True)
    return


def sum_games(data: str, average: bool, batched_num_games: int, batched: bool = False)-> Tuple[np.ndarray, int]:
    '''
    Iterate over each file in the specified data filepath, and calculates the sum (or the average).

    Arguments:
        - data (str): the filepath to the specified data folder
        - average (bool): if True, returns the average (by dividing by the number of files in the directory)
    
    Output:
        - games_total (numpy.ndarray): a numpy array that either contains:
            - the average of the files if average is True
            - the sum of the files if average is False
        - num_games (int): the number of games played
    '''
    files = [file for file in os.listdir(data) if os.path.isfile(os.path.join(data, file))] # iterate through /data directory, only process files
    games_total = None # where the sum of the games is going
    for file in files:
        file_path = os.path.join(data,file) # get file name and directory
        game = np.load(file_path, allow_pickle=True) # load the file
        if games_total is None:
            games_total = game # initialize games_total sum array
        else:
            games_total += game
    if batched:
        num_games = batched_num_games
    else:
        num_games = len(files)
    if average:
        return np.divide(games_total, num_games), num_games
    return games_total, num_games # divide each individual element by the number of games played
