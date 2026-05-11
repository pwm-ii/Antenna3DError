import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm, colors

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
COORD_ROUNDING = 3  # Decimal places for Phi/Theta alignment

# Original File Column Names
ORIG_TARGET = 'dB(GainTotal)'
ORIG_PHI    = 'Phi[deg]'
ORIG_THETA  = 'Theta[deg]'

# Interpolated File Column Names
INTERP_TARGET = 'Gain[dB]'
INTERP_PHI    = 'Phi[deg]'
INTERP_THETA  = 'Theta[deg]'

# DATA FORMATTING NOTES:
# 1. Theta (Elevation) is expected in range [0, 180]
# 2. Phi (Azimuth) is expected in range [0, 360] or [-180, 180]
# 3. CSVs should be long-form (one row per coordinate point)
# ==========================================

class PlotPanel(tk.Frame):
    def __init__(self, parent, title, projection=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111, projection=projection)
        self.ax.set_title(title, fontsize=10, pad=20)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

class AntennaComparisonViewer:
    def __init__(self, df_merged, mse, rmse, mean_bias):
        self.root = tk.Tk()
        self.root.title(f"Antenna Pattern Comparison")
        self.root.geometry("1500x700")
        
        # Pivot Data
        self.grid_interp = df_merged.pivot(index=ORIG_PHI, columns=ORIG_THETA, values="val_interp")
        self.grid_orig   = df_merged.pivot(index=ORIG_PHI, columns=ORIG_THETA, values="val_orig")
        self.grid_error  = df_merged.pivot(index=ORIG_PHI, columns=ORIG_THETA, values='abs_error')

        # Coordinates
        self.phi_coords = np.deg2rad(self.grid_interp.index.values)
        self.theta_coords = self.grid_interp.columns.values 
        
        # Pattern Scaling
        self.vmin = min(self.grid_interp.min().min(), self.grid_orig.min().min())
        self.vmax = max(self.grid_interp.max().max(), self.grid_orig.max().max())

        self._setup_ui(mse, rmse, mean_bias)

    def _setup_ui(self, mse, rmse, mean_bias):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Plot 1: Interpolated
        self.p1 = PlotPanel(self.root, "Reconstructed Pattern (Interpolated)", projection='polar')
        self.p1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._draw_polar_heatmap(self.p1, self.grid_interp.values, cm.nipy_spectral, "Gain [dB]", self.vmin, self.vmax)

        # Plot 2: Original
        self.p2 = PlotPanel(self.root, "Actual Pattern (Original)", projection='polar')
        self.p2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self._draw_polar_heatmap(self.p2, self.grid_orig.values, cm.nipy_spectral, "Gain [dB]", self.vmin, self.vmax)

        # Plot 3: Error Heatmap (Legend Inverted)
        self.p3 = PlotPanel(self.root, "Absolute Error Heatmap", projection='polar')
        self.p3.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        # Using YlOrRd_r so 0 dB is white/yellow and 15 dB is dark red
        self._draw_polar_heatmap(self.p3, self.grid_error.values, cm.viridis_r, "Abs Error [dB]", invert_cbar=True)

        # Stats Bar
        stats_frame = ttk.Frame(self.root)
        stats_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        bias_desc = "Optimistic" if mean_bias > 0 else "Conservative"
        lbl_text = (f"MSE: {mse:.4f}  |  RMSE: {rmse:.4f}  |  "
                    f"Mean Bias: {mean_bias:.4f} dB ({bias_desc})")
        ttk.Label(stats_frame, text=lbl_text, font=("Arial", 11, "bold")).pack()

    def _draw_polar_heatmap(self, panel, data, cmap, label, vmin=None, vmax=None, invert_cbar=False):
        phi_grid, theta_grid = np.meshgrid(self.phi_coords, self.theta_coords, indexing='ij')

        mesh = panel.ax.pcolormesh(
            phi_grid, theta_grid, data,
            cmap=cmap,
            shading='auto',
            vmin=vmin if vmin is not None else np.nanmin(data),
            vmax=vmax if vmax is not None else np.nanmax(data)
        )

        panel.ax.set_theta_zero_location('N') 
        panel.ax.set_theta_direction(-1)      
        panel.ax.set_rlim(0, 180)             
        panel.ax.set_rticks([0, 45, 90, 135, 180])
        panel.ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°'], fontsize=7)
        
        cbar = panel.figure.colorbar(mesh, ax=panel.ax, label=label, shrink=0.7, pad=0.1)
        if invert_cbar:
            cbar.ax.invert_yaxis()  # Puts minimal error (0) at the top

    def show(self):
        self.root.mainloop()

def calculate_antenna_mse(file_interp, file_orig):
    try:
        df_interp = pd.read_csv(file_interp)
        df_orig = pd.read_csv(file_orig)
    
        df_interp.columns = [c.strip() for c in df_interp.columns]
        df_orig.columns = [c.strip() for c in df_orig.columns]
        
        df_orig[ORIG_PHI] = df_orig[ORIG_PHI].round(COORD_ROUNDING)
        df_orig[ORIG_THETA] = df_orig[ORIG_THETA].round(COORD_ROUNDING)
        df_interp[INTERP_PHI] = df_interp[INTERP_PHI].round(COORD_ROUNDING)
        df_interp[INTERP_THETA] = df_interp[INTERP_THETA].round(COORD_ROUNDING)
                
        merged = pd.merge(
            df_orig[[ORIG_PHI, ORIG_THETA, ORIG_TARGET]], 
            df_interp[[INTERP_PHI, INTERP_THETA, INTERP_TARGET]], 
            left_on=[ORIG_PHI, ORIG_THETA],
            right_on=[INTERP_PHI, INTERP_THETA]
        )

        if merged.empty:
            print("Error: No matching coordinates found.")
            return

        merged = merged.rename(columns={ORIG_TARGET: "val_orig", INTERP_TARGET: "val_interp"})
        merged['diff'] = merged["val_interp"] - merged["val_orig"]
        merged['sq_error'] = merged['diff'] ** 2
        merged['abs_error'] = merged['diff'].abs() 
        
        app = AntennaComparisonViewer(merged, merged['sq_error'].mean(), (merged['sq_error'].mean())**0.5, merged['diff'].mean())
        app.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path_interp = r"C:\User\Downloads\ApproximationFile.csv"
    path_orig   = r"C:\User\Downloads\OriginalFile.csv"
    calculate_antenna_mse(path_interp, path_orig)
