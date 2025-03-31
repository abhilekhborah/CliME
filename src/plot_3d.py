import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

MODEL_NAME = "Model Name"

def load_climate_data(file_path):
    """
    Load and process climate messaging data.
    Returns a DataFrame with CAQ dimensions.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file: {e}")
        print("Attempting alternative parsing method...")
        try:
            with open(file_path, 'r') as file:
                text = file.read()
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx >= 0 and end_idx > 0:
                    data = json.loads(text[start_idx:end_idx])
                else:
                    print("Could not identify valid JSON structure in file")
                    return pd.DataFrame()
        except Exception as e2:
            print(f"Secondary parsing attempt failed: {e2}")
            return pd.DataFrame()
    
    results = []
    for msg_id, msg_data in data.items():
        if not isinstance(msg_data, dict):
            continue
        msg_text = msg_data.get('text', f"Message {msg_id}")
        try:
            actionability_caq = float(msg_data.get('actionability', {}).get('caq', np.nan))
            criticality_caq = float(msg_data.get('criticality', {}).get('caq', np.nan))
            justice_caq = float(msg_data.get('justice', {}).get('caq', np.nan))
            results.append({
                'message_id': msg_id,
                'message_text': msg_text[:50] + '...' if len(msg_text) > 50 else msg_text,
                'actionability_caq': actionability_caq,
                'criticality_caq': criticality_caq,
                'justice_caq': justice_caq
            })
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Warning: Could not extract CAQ scores for message {msg_id}: {e}")
    
    df = pd.DataFrame(results)
    before_count = len(df)
    df = df.dropna(subset=['actionability_caq', 'criticality_caq', 'justice_caq'])
    after_count = len(df)
    if before_count > after_count:
        print(f"Note: Removed {before_count - after_count} messages with missing data")
    return df

def create_enhanced_3d_plot(df, figsize=(12, 10), 
                            save_path=f'{MODEL_NAME.replace(" ", "_")}_3d_caq_plot.png'):
    """
    Create a 3D scatter plot of CAQ dimensions.
    Each message is plotted as a point in 3D space, and the mean with error bars is highlighted.
    """
    # Calculate statistics
    mean_action = df['actionability_caq'].mean()
    mean_criticality = df['criticality_caq'].mean()
    mean_justice = df['justice_caq'].mean()
    
    std_action = df['actionability_caq'].std()
    std_criticality = df['criticality_caq'].std()
    std_justice = df['justice_caq'].std()
    
    overall_mean = np.mean([mean_action, mean_criticality, mean_justice])
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=figsize, facecolor='#f8f8f8')
    ax = fig.add_subplot(111, projection='3d', facecolor='#f9f9f9')
    
    sc = ax.scatter(df['actionability_caq'], df['criticality_caq'], df['justice_caq'], 
                    c=df['actionability_caq'], cmap='viridis', alpha=0.6, edgecolor='k')
    
    # Plot the mean point
    ax.scatter([mean_action], [mean_criticality], [mean_justice], 
               color='#e74c3c', s=150, marker='*', label='Mean CAQ')
    
    # Add error bars for the mean (one for each axis)
    ax.plot([mean_action - std_action, mean_action + std_action], 
            [mean_criticality, mean_criticality], 
            [mean_justice, mean_justice], color='#e74c3c', lw=2)
    ax.plot([mean_action, mean_action], 
            [mean_criticality - std_criticality, mean_criticality + std_criticality], 
            [mean_justice, mean_justice], color='#e74c3c', lw=2)
    ax.plot([mean_action, mean_action], 
            [mean_criticality, mean_criticality], 
            [mean_justice - std_justice, mean_justice + std_justice], color='#e74c3c', lw=2)
    
    # Set axis labels and title
    ax.set_xlabel("Actionability", fontsize=12, fontweight='bold')
    ax.set_ylabel("Criticality", fontsize=12, fontweight='bold')
    ax.set_zlabel("Justice", fontsize=12, fontweight='bold')
    # ax.set_title(f'{MODEL_NAME}: CAQ Dimensions Analysis', fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    
    # Add a colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("CAQ", fontsize=12)
    
    # Annotation with summary statistics
    stats_text = (
        f"Mean CAQ Values:\n"
        f"Actionability: {mean_action:.3f} ± {std_action:.3f}\n"
        f"Criticality: {mean_criticality:.3f} ± {std_criticality:.3f}\n"
        f"Justice: {mean_justice:.3f} ± {std_justice:.3f}\n"
        f"Overall Mean: {overall_mean:.3f}\n"
        f"Descriptors: {len(df)}"
    )
    ax.text2D(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.view_init(elev=30, azim=30)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D CAQ plot saved as '{save_path}'")
    
    return fig

if __name__ == "__main__":
    file_path = "/Users/deltae/Downloads/AIISC/ACL - NLPforPosImp/combined_llama70_eval.json"
    
    dir ="/scratch/user/hasnat.md.abdullah/Climmeme/outputs/revised_caq"

    # Traverse all JSON files in the directory
    json_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.json')]

    for file_path in json_files:
        print(f"Processing file: {file_path}")
        df = load_climate_data(file_path)
        
        if df.empty:
            print(f"Error: No valid data found in the file {file_path}.")
        else:
            print(f"Successfully loaded {len(df)} messages with CAQ dimensions from {file_path}.")
            print("Creating enhanced 3D CAQ plot...")
            create_enhanced_3d_plot(df, save_path=f"{os.path.splitext(file_path)[0]}_3d_caq_plot.png")
            print("Done!")
    # print(f"Loading data from {file_path}...")
    # df = load_climate_data(file_path)
    
    # if df.empty:
    #     print("Error: No valid data found in the file.")
    # else:
    #     print(f"Successfully loaded {len(df)} messages with CAQ dimensions.")
    #     print("Creating enhanced 3D CAQ plot...")
    #     create_enhanced_3d_plot(df)
    #     print("Done!")
