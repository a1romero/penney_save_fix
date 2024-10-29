import itertools
import pandas as pd
import numpy as np
import os
import src.processing as processing
import src.visualization as visualization
import json
from typing import Tuple
import matplotlib.pyplot as plt
import mpld3
from matplotlib.ticker import PercentFormatter

def shuffle_deck(seed:int) -> str:
    '''Generates a single shuffled deck of 0s and 1s. 

    Arguments: 
    seed (int): Seed to ensure reproducibility

    Output:
    A string of 52 characters where each character is either '0' or '1', representing a shuffled deck.
    '''
    rng = np.random.default_rng(seed = seed) 
    deck = np.ndarray.flatten((np.stack((np.ones(26), np.zeros(26)), axis= 0).astype(int))) # make a deck of 26 ones and zeroes
    rng.shuffle(deck) # then shuffle the deck
    return ''.join(map(str, deck)) # convert to string

def results_for_viz(x):
    """
    Takes in results from play_n_games() function. Reformats results of simulations for heatmap visualization.

    Arguments:
    x (dict): A dictionary containing the results from the play_n_games() function. 
                Keys:
                - 'cards': A list of number of cards won by the Me player.
                - 'tricks': A list of number of tricks won by the Me player.
                - 'cards_ties': A list of counts of card ties.
                - 'tricks_ties': A list of counts of trick ties.

    Output:
    Saves the reformatted results to a JSON file named 'results.json' in the 'results' folder. If the 
    folder does not exist, it will be created.
    """
    x['cards'] = x['cards'].tolist()
    x['tricks'] = x['tricks'].tolist()
    x['cards_ties'] = x['cards_ties'].tolist()
    x['tricks_ties'] = x['tricks_ties'].tolist()

    data_folder = 'results'
    # make results folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    with open(os.path.join(data_folder,'results.json'), 'w') as json_file:
        json.dump(x, json_file, indent=4) # dump results into json file

def play_n_games(n: int, data: str, initial_seed: int = 0, batched: bool = False, batched_num_games: int = 0) -> dict:
    """
    Runs simulations for n games, saving results to the specified data folder.
    
    Arguments:
    - n (int): Number of games to simulate.
    - data (str): Folder to store results for each iteration.
    - initial_seed (int): Starting seed value for random number generation.

    Output:
    Saves the results of the simulations in a JSON file named 'results.json' within the specified data folder. 
    The file contains information about each game's outcome, including details such as number of cards won, number 
    of tricks won, and ties that occurred during the games for both variations. If the data folder does 
    not exist, it will be created. The results are added to the results that already exist in the results.json file. 
    """
    # makes data folder if it does not exist
    if not os.path.exists(data):
        os.makedirs(os.path.join(data,'cards'))
        os.makedirs(os.path.join(data,'cards_ties'))
        os.makedirs(os.path.join(data,'tricks'))
        os.makedirs(os.path.join(data,'tricks_ties'))

    if batched: 
        batch_cards = np.empty((8,8))
        batch_tricks = np.empty((8,8))
        batch_cards_ties = np.empty((8,8))
        batch_tricks_ties = np.empty((8,8))

        for i in range(n): # runs n games
            deck = shuffle_deck(seed=initial_seed + i)
            i_cards, i_tricks, i_cards_ties, i_tricks_ties = processing.play_one_deck(data = data, deck = deck, batched = batched)

            batch_cards += i_cards
            batch_tricks += i_tricks
            batch_cards_ties += i_cards_ties
            batch_tricks_ties += i_tricks_ties

        processing.batch_save(batch_cards = batch_cards,
                              batch_tricks = batch_tricks,
                              batch_cards_ties = batch_cards_ties,
                              batch_tricks_ties = batch_tricks_ties,
                              name = initial_seed,
                              data = data)
    else:
        for i in range(n):
            deck = shuffle_deck(seed = initial_seed + i)
            processing.play_one_deck(data = data, deck = deck, batched = batched)

    filename = ['cards', 'cards_ties', 'tricks', 'tricks_ties']
    results = {}
    n_games = []

    # calculate the average for each folder
    for folder in filename:
        results[folder], g_num = processing.sum_games(f'{data}/{folder}', True, batched = batched, batched_num_games = batched_num_games,)
        n_games.append(g_num)
    results['n'] = n_games[0]
    # Reformatting and save results for viz
    results_for_viz(results)
    return results

def save_figures(figures, file_name):
    """
    Save figures as png images and embed them in an HTML file.
    
    Arguments:
    - figures: the matplotlib figure object to save 
    - file_name (str): The name you want to give to the html and png files

    Output:
    Saves the figure as a png image in the 'figures' folder and creates an HTML file that embeds this figure. 
    The png and html files are named according to the specified file_name with the corresponding extension.
    """
    # saving figure as png in 'figures' folder
    filename = file_name.replace('.html', '.png')
    figures.savefig(filename, bbox_inches='tight')

    # creating HTML
    html_content = html_content = mpld3.fig_to_html(figures)

    # saving the HTML to a file
    with open(file_name, 'w') as f:
        f.write(html_content)


def create_heatmap(variation: str, ax: plt.Axes = None, hide_y: bool = False, pkg: bool = False) -> [plt.Figure, plt.Axes]:
    '''
    Creates a heatmap visualization based on either card or trick data, saving the output as a html and png file.
    Visualization specifications are based on class specifications. 

    Arguments:
        - variation (str): Specifies the type of data to visualize. Use 'cards' for card variation or 
            'tricks' for trick variation.
        - ax (plt.Axes, optional): Matplotlib Axes object where the heatmap will be drawn. If None, 
            a new figure and axes will be created.
        - hide_y (bool, optional): If True, hides the y-axis of the heatmap. Default is False.
        - pkg (bool, optional): If False, creates and saves heatmaps as individual figures for standalone use. 
            Default is False.

    Output:
        A tuple containing the Matplotlib figure and axes objects for the created heatmap. 
        If `pkg` is `False`, the figure is saved as a PNG image in the 'figures' folder and creates an 
        HTML file that embeds this figure. The files are named according to the specified variation.
    '''

    # make figures directory if it does not exist
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # format the data from results.json
    data = visualization.get_data()
    
    # for cards and tricks, generate variation-specific data for the heatmaps
    if variation == 'cards':
        t1_data = visualization.format_data(np.array(data['cards']), countwins=True)
        ties_data = visualization.format_data(np.array(data['cards_ties']), countwins=True)
        title = "My Chance of Winning (By Cards)"
        filename = 'figures/cards_heatmap.html'
    elif variation == 'tricks':
        t1_data = visualization.format_data(np.array(data['tricks']), countwins=True)
        ties_data = visualization.format_data(np.array(data['tricks_ties']), countwins=True)
        title = "My Chance of Winning (By Tricks)"
        filename = 'figures/tricks_heatmap.html'
    else:
        raise ValueError("Invalid data_type specified. Use 'cards' or 'tricks'.")

    # make annotations with cards and ties
    annots = visualization.make_annots(t1_data, ties_data)

    # make the heatmap with all of the data
    fig, ax = visualization.make_heatmap(data=t1_data, annots=annots, title=title, n=data['n'], cbar_single=False, ax=ax, hide_y=hide_y)
    ax.set_aspect('equal', adjustable='box')

    # changes for a single visualization
    if not pkg:
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        save_figures(fig, filename)

    return fig, ax

def make_heatmap_package() -> [plt.Figure, plt.Axes]:
    '''
    Create a 1x2 grid of heatmaps based on the given data, with shared colorbar.
    This function generates two heatmaps side by side: one for card data and one for trick data.
    The heatmaps are created using the `create_heatmap` function.

    Returns:
        A tuple containing the Matplotlib figure and axes objects for the created heatmaps. The figure is 
        saved as an HTML file and a PNG image in the 'figures' directory with the filenames 'pkg_heatmap.html' 
        and 'pkg_heatmap.png', respectively.
    '''
    
    # make the heatmap space
    fig, ax = plt.subplots(1, 2, 
                           figsize=(8*2, 8), 
                           gridspec_kw={'wspace': 0.05})
    
    # make heatmaps for tricks and cards, they should map onto the previously created fig and ax
    create_heatmap('cards', ax=ax[0],pkg=True)
    create_heatmap('tricks',ax=ax[1], hide_y=True,pkg=True)
    
    # create one colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  
    colorbar = fig.colorbar(ax[0].collections[0], cax=cbar_ax)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # make a folder for figures if it does not exist already
    if not os.path.exists('figures'):
        os.makedirs('figures')
    save_figures(fig, 'figures/pkg_heatmap.html')
    fig.savefig('figures/pkg_heatmap.png', bbox_inches='tight')
    return fig, ax

def score_deck(deck: str,
               seq1: str,
               seq2: str) -> Tuple[int]:
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
