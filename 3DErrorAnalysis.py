import pandas as pd
import numpy as np
import sys
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm

class PlotPanel(tk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 5), dpi=100)
        self.figure.set_tight_layout(True)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title, fontsize=10)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

class AntennaComparisonViewer:
    def __init__(self, grid_interp, grid_orig, grid_error, mse, rmse, mean_bias):
        self.root = tk.Tk()
        self.root.title(f"Antenna Pattern Comparison (MSE: {mse:.4f})")
        self.root.geometry("1400x600")
        
        self.grid_interp = grid_interp
        self.grid_orig = grid_orig
        self.grid_error = grid_error
        
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # Extent for Plots: Theta (x) 0..180, Phi (y) 0..360
        self.extent = [0, 180, 0, 360] 

        # Plot 1: Reconstructed Pattern
        self.p1 = PlotPanel(self.root, "Reconstructed Pattern (Interpolated)")
        self.p1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._draw_heatmap(self.p1, self.grid_interp, cmap=cm.nipy_spectral, label="Gain [dB]")

        # Plot 2: Original 
        self.p2 = PlotPanel(self.root, "Actual Pattern (Original)")
        self.p2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self._draw_heatmap(self.p2, self.grid_orig, cmap=cm.nipy_spectral, label="Gain [dB]")

        # Plot 3: Total Error
        self.p3 = PlotPanel(self.root, "Absolute Error")
        self.p3.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self._draw_heatmap(self.p3, self.grid_error, cmap=cm.jet, label="Abs Error [dB]")

        stats_frame = ttk.Frame(self.root)
        stats_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        
        if mean_bias > 0:
            bias_desc = "Optimistic"
        else:
            bias_desc = "Conservative"

        lbl_text = (f"COMPARISON STATISTICS  |  "
                    f"MSE: {mse:.4f}  |  "
                    f"RMSE: {rmse:.4f}  |  "
                    f"Mean Bias: {mean_bias:.4f} dB ({bias_desc})")
        
        ttk.Label(stats_frame, text=lbl_text, font=("Arial", 11, "bold")).pack()

    def _draw_heatmap(self, panel, data, cmap, label):
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        im = panel.ax.imshow(
            data,
            extent=self.extent,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='bilinear',
            vmin=vmin, vmax=vmax
        )
        
        panel.ax.set_xlabel("Theta (degree)")
        panel.ax.set_ylabel("Phi (degree)")
        panel.figure.colorbar(im, ax=panel.ax, label=label)

    def show(self):
        self.root.mainloop()

def calculate_antenna_mse(file_interp, file_orig):
    # Alignment determined based on 'Phi[deg]' and 'Theta[deg]' columns.
    
    print(f"Loading Interpolated Data: {file_interp}")
    print(f"Loading Original Data:     {file_orig}")

    try:
        df_interp = pd.read_csv(file_interp)
        df_orig = pd.read_csv(file_orig)
    
        df_interp.columns = [c.strip() for c in df_interp.columns]
        df_orig.columns = [c.strip() for c in df_orig.columns]
        
        target_col = 'dB10normalize(GainTotal)' #Change as needed, hardcoded data column name
        req_cols = ['Phi[deg]', 'Theta[deg]', target_col]
        
        for df, name in [(df_interp, "Interpolated"), (df_orig, "Original")]:
            if not all(col in df.columns for col in req_cols):
                print(f"Error: {name} file missing required columns: {req_cols}")
                print(f"Available columns: {list(df.columns)}")
                return
                
        merged = pd.merge(
            df_orig, 
            df_interp, 
            on=['Phi[deg]', 'Theta[deg]'], 
            suffixes=('_orig', '_interp')
        )

        if merged.empty:
            print("Error: No matching Phi/Theta coordinates found between files.")
            print("Check if angle ranges (e.g. -180 vs 0..360) match.")
            return

        print(f"Aligned {len(merged)} data points for comparison.")

        col_orig = f"{target_col}_orig"
        col_interp = f"{target_col}_interp"
        
        merged['diff'] = merged[col_interp] - merged[col_orig]
        merged['sq_error'] = merged['diff'] ** 2
        merged['abs_error'] = merged['diff'].abs() 
        
        mse = merged['sq_error'].mean()
        rmse = mse ** 0.5
        mean_bias = merged['diff'].mean()
        
        print("-" * 40)
        print(f"Mean Squared Error (MSE):      {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        
        # Bias Report
        if mean_bias > 0:
            print(f"Mean Bias:                     {mean_bias:.6f} dB (Optimistic)")
        else:
            print(f"Mean Bias:                     {mean_bias:.6f} dB (Conservative)")
        print("-" * 40)

        print("\nTop 5 Largest Differences:")
        print(merged.nlargest(5, 'sq_error')[['Phi[deg]', 'Theta[deg]', col_orig, col_interp, 'diff']])

        print("\nPreparing Visualization...")

        grid_interp = merged.pivot(index='Phi[deg]', columns='Theta[deg]', values=col_interp)
        grid_orig = merged.pivot(index='Phi[deg]', columns='Theta[deg]', values=col_orig)
        grid_error = merged.pivot(index='Phi[deg]', columns='Theta[deg]', values='abs_error')
        
        grid_interp.sort_index(axis=0, inplace=True) 
        grid_interp.sort_index(axis=1, inplace=True)
        
        grid_orig = grid_orig.reindex_like(grid_interp)
        grid_error = grid_error.reindex_like(grid_interp)

        app = AntennaComparisonViewer(
            grid_interp.values, 
            grid_orig.values, 
            grid_error.values, 
            mse, 
            rmse,
            mean_bias
        )
        app.show()

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # USER INPUT: Enter Filepath Below ENTER FILEPATH BELOW
    file_path_interpolated = r"C:\Users\Username\Downloads\3DInterpolatedSummingPyramid.csv"
    file_path_original = r"C:\Users\Username\Downloads\3DPolarPlotAntenna.csv"

    if not os.path.exists(file_path_interpolated) or not os.path.exists(file_path_original):
        print("Please update the file paths in the script to point to your CSV files.")
        print(f"Looking for: {file_path_interpolated}")
    else:

        calculate_antenna_mse(file_path_interpolated, file_path_original)
