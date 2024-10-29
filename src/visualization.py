import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# Settings for the individual figures
FIG_HIGH = 8
FIG_WIDE = 8

TITLE_SIZE = 18
LABEL_SIZE = 14
TICKLABEL_SIZE = 12
TICK_SIZE = 10
ANNOT_SIZE = 8


def get_data() -> dict:
    '''
    Load the results.json file, and return a dictionary representing the contents of the file.
    
    Arguments: None

    Output:
        - x (dict): dict of results from the results.json folder.
    '''
    file_path = os.path.join('results', 'results.json')

    # Load the JSON file
    with open(file_path, 'r') as json_file:
        x = json.load(json_file)

    return x

def format_data(array: np.ndarray, countwins= False) -> np.ndarray:
    '''
    Properly formats the input data, by rounding the numbers in the array and returning an array of percentages, as whole numbers, or probabilities, as decimals.
    Additionally, it fills the diagonal with np.nan (for proper representation in create_heatmap functions in main).
    
    Arguments:
        - array (np.ndarray): the array of probabilities to reformat
        - countwins (bool): if True, return the whole number structure (as percentages out of 100). If False, return just the decimals.

    Output:
        - formatted (np.ndarray): A 2D NumPy array that has been rounded and formatted. The diagonal from the bottom-left to the top-right is filled with np.nan
    '''

    array = np.array(array, dtype=float)  # Ensure input is a NumPy array
    # Return whole number structure
    if countwins:
        temp = np.round(array * 100, 0)
    # Return the decimals
    else:
        temp = np.round(array, 2)

    # Fill the diagonal with NaN
    formatted = fill_diag(temp, np.nan)
    return formatted

def fill_diag(array: np.ndarray, filler) -> np.ndarray:
    '''
    Fills the diagonal from the bottom-left to the top-right of array with a specified filler object.

    Arguments:
        - array (np.ndarray): the array to be properly formatted and filled with the filler
        - filler (Any): object to insert in the diagonal, likely None, nan, or 0

    Output:
        - array with diagonal filled in with filler
    '''
    
    array = np.array(array)  # Ensure input is a NumPy array
    flipped_array = np.flipud(array) # # Flipping upside down
    np.fill_diagonal(flipped_array, filler) # Filling the diagonal
    return np.flipud(flipped_array) # flip back to original orientation


def make_annots(wins : np.ndarray,ties: np.ndarray) -> np.ndarray:
    '''
    Iterates through provided wins and ties to return an array formatted like this: wins (ties). Used to create the heatmap annotations.

    Arguments:
        - wins (np.ndarray): the wins for the variation formatted using format_data
        - ties (np.ndarray): the ties for the variation formatted using format_data

    Output:
        - array where each position is in the form win (tie) for each specification.
    '''
    annots = []
    for i in range(8):
        row = []
        for j in range(8):
            # Format string labels
            row.append(f'{str(wins[i,j])[:-2]} ({str(ties[i,j])[:-2]})')
        annots.append(row)
    # Fill diagonal cells with filler object
    annots=fill_diag(annots, "")
    return np.array(annots)

def make_heatmap(data: np.ndarray,
                 annots: np.ndarray,
                 title:str,
                 n: int,
                 hide_y: bool = False,
                 cbar_single: bool = True,
                 ax: plt.Axes = None
                ) -> [plt.Figure , plt.Axes]:
    '''
    If ax is None, create a new figure.
    Otherwise, add the heatmap to the provided ax.

    Returns a heatmap with the input data.

    Arguments:
        - data (np.ndarray): the win data for the heatmap
        - annots (np.ndarray): the annotations that will appear on the heatmap (allowing a representation of both wins and ties in one heatmap)
        - title (str): title of the heatmap (specified in main.py for the different variations)
        - n (int): number of games in the dataset
        - hide_y (bool): for make_heatmap_package, hide the y axis on the first one so there is no unnecessary repetition
        - cbar_single (bool): specifies whether to add a colorbar (for single plots)
        - ax (plt.Axes): adds the heatmap to a specified ax. If ax is None, makes a new figure.

    Output:
        - fig (plt.Figure): Figure with the created heatmap
        - ax (plt.Axes): ax with the created heatmap
    '''
    
    if ax is None:
        # Create new figure
        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDE, FIG_HIGH))
    else:
        # Get parent figure
        fig = ax.get_figure()

    # Tick labels as letters
    seqs= ['BBB','BBR','BRB','BRR','RBB','RBR','RRB','RRR'] 

    # Heatmap properties
    settings = {
        'vmin': 0,
        'vmax': 100,
        'linewidth': .5,
        'cmap': 'Blues',
        'cbar': False,
        'annot': annots,
        'annot_kws': {"size": ANNOT_SIZE},
        'fmt': ''
    }

    # Create the heatmap using seaborn
    sns.heatmap(data=data, ax=ax,  **settings)
    # Set x and y labels
    ax.set_xlabel('Me', fontsize=LABEL_SIZE)
    ax.set_ylabel('Opponent', fontsize=LABEL_SIZE)
    # Set the tick labels for x-axis and y-axis
    ax.set_xticklabels(seqs, fontsize=TICK_SIZE)
    ax.set_yticklabels(seqs[::-1], fontsize=TICK_SIZE)
    ax.set_facecolor('#DBDBDB')

    # Add a color bar if prompted
    if cbar_single: 
        cbar_ax = fig.add_axes([.95, 0.11, 0.035, .77])
        cb = fig.colorbar(ax.collections[0], cax=cbar_ax)
        # Adjusting the tickmark sizes on color bar 
        cb.ax.tick_params(labelsize=TICK_SIZE)
        cb.outline.set_linewidth(.2)
        
    # Set the title of the visualization
    ax.set_title(title+'\n(n='+str(n)+')', fontsize=TITLE_SIZE)
                    
    # For package heatmap visualization, both the y-ticks and axis title should be hidden on 2nd subplot
    if hide_y:
        ax.set_yticks([])
        ax.set_ylabel(None)
     
    return fig, ax
