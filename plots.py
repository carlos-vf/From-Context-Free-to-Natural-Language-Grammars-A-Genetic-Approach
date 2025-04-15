import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource



def read_txt_files_to_dataframe(folder_path, key):
    # Find all txt files in the given folder
    txt_files = glob.glob(f"{folder_path}/*.txt")
    
    dataframes = {}
    for file in txt_files:
        filename = file.split('\\')[-1][0:-4]  # Extract file name without extension
        parts = filename.split('_')  # Split filename by '_'
        language, nonterminals, elite, bloat = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
        
        df = pd.read_csv(file, sep='\t')  # Read file as a DataFrame
        df['language'] = language  # Add language column
        df['nonterminals'] = nonterminals  # Add nonterminals column
        df['elite'] = elite  # Add elite column
        df['bloat'] = bloat  # Add bloat column
        
        dataframes[locals()[key]] = df
    
    return dataframes


def plot_3d_hillshade(dataframes, variable_x, variable_z, cmap, lang):
    # Combine all data into a grid
    all_x, all_y, all_z = [], [], []
    
    for key, df in dataframes.items():
        x_values = df[variable_x].values  # Selected variable as X
        all_x.extend(x_values)
        all_y.extend(df['iter'].values)  # Iteration as Y
        all_z.extend(df[variable_z].values)  # Avg fitness as Z
    
    x = np.array(all_x)
    y = np.array(all_y)
    z = np.array(all_z)
    
    # Create meshgrid
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)
    z_grid = np.zeros_like(x_grid)
    
    for i in range(len(x)):
        x_idx = np.where(x_unique == x[i])[0][0]
        y_idx = np.where(y_unique == y[i])[0][0]
        z_grid[y_idx, x_idx] = z[i]
    
    # Set up plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    ls = LightSource(270, 45)
    rgb = ls.shade(z_grid, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    
    ax.set_xlabel(f"{variable_x.capitalize()} (X)")
    ax.set_ylabel("Iteration (Y)")
    ax.set_zlabel(f"{variable_z.capitalize()} (Z)")
    ax.set_title(f"{variable_x.capitalize()} vs. {variable_z.capitalize()}")
    plt.savefig(f"results/plots/{lang}/{variable_x}_{variable_z}_plot.png")



for lang in ["eng", "esp"]:

    # Elite variation
    folder_path = f"results/{lang}/elite_variation" 
    dataframes = read_txt_files_to_dataframe(folder_path, key="elite")
    plot_3d_hillshade(dataframes, variable_x="elite", variable_z='avg_fitness', cmap=cm.gist_earth, lang=lang) 
    plot_3d_hillshade(dataframes, variable_x="elite", variable_z='avg_size', cmap=cm.magma, lang=lang) 

    # Bloat variation
    folder_path = f"results/{lang}/bloat_variation" 
    dataframes = read_txt_files_to_dataframe(folder_path, key="bloat")
    plot_3d_hillshade(dataframes, variable_x="bloat", variable_z='avg_fitness', cmap=cm.gist_earth, lang=lang) 
    plot_3d_hillshade(dataframes, variable_x="bloat", variable_z='avg_size', cmap=cm.magma, lang=lang) 

    # Nonterminal variation
    folder_path = f"results/{lang}/nonterminals_variation" 
    dataframes = read_txt_files_to_dataframe(folder_path, key="nonterminals")
    plot_3d_hillshade(dataframes, variable_x="nonterminals", variable_z='avg_fitness', cmap=cm.gist_earth, lang=lang) 
    plot_3d_hillshade(dataframes, variable_x="nonterminals", variable_z='avg_size', cmap=cm.magma, lang=lang) 
