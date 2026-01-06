===========================================================================
                    ANTENNA PATTERN ERROR ANALYSIS TOOL
                                 USER GUIDE
===========================================================================

1. PURPOSE
---------------------------------------------------------------------------
Verifies the accuracy of 3D antenna pattern interpolation by comparing a 
Predicted CSV against a Ground Truth CSV.


2. HOW TO RUN
---------------------------------------------------------------------------
1. Open Script
2. Set Paths: Update the file paths at the bottom of the script:
     - file_path_interpolated: Your generated/predicted CSV.
     - file_path_original: The reference/measured CSV.
3. Run script


3. INPUT REQUIREMENTS
---------------------------------------------------------------------------
Both CSV files must share the same angular grid (step size) and contain 
these exact headers:

  1. Phi[deg]
  2. Theta[deg]
  3. dB10normalize(GainTotal) 
     (Rename your gain column to this if necessary)


4. INTERPRETING RESULTS
---------------------------------------------------------------------------
RMSE (Root Mean Sq Error):
   - The primary accuracy metric.
   - < 1.0 dB: Excellent match.
   - > 3.0 dB: Mismatch (likely missing side lobes).

Mean Bias:
   - The average tendency of the prediction.
   - Positive (+): Optimistic (Predicting higher signal than reality).
   - Negative (-): Conservative (Predicting lower signal than reality).

Error Heatmaps:
   - Left/Center: Visual comparison of patterns.
   - Right (Error): 
       - Blue/Black: Good (low error).
       - Red/Yellow: Indicates a mismatch.
