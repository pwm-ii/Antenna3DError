ANTENNA PATTERN ERROR ANALYSIS TOOL
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1.) PURPOSE
---------------------------------------------------------------------------
This tool provides a visual and statistical comparison between two 3D antenna 
radiation patterns. It is designed to evaluate the accuracy of interpolation algorithms.


2.) INPUT REQUIREMENTS
---------------------------------------------------------------------------
The script performs an automated join based on angular coordinates. To ensure 
successful alignment, both files should use the same grid resolution. Script 
normalizes gain to max value for both files. 

Define CSV Headers:
   e.g. Original Truth File:
   - Phi[deg]
   - Theta[deg]
   - dB(GainTotal)

   e.g. Interpolated File:
   - Phi[deg]
   - Theta[deg]
   - Gain[dB]

Data Formatting Notes:
   - Theta (Elevation): Expected range [0, 180].
   - Phi (Azimuth): Expected range [0, 360] or [-180, 180].
   - The tool automatically rounds coordinates to 3 decimal places to 
     mitigate floating-point mismatches during the merge process.


3.) METRICS & RESULTS
---------------------------------------------------------------------------
The tool calculates three primary statistical indicators:

MSE (Mean Squared Error):
   - Measures the average squared difference between predicted and actual gain.
   - Useful for penalizing large individual outliers.

RMSE (Root Mean Square Error):
   - The primary accuracy metric, expressed in the same units as the input (dB).
   - Represents the standard deviation of the prediction errors.

Mean Bias:
   - Indicates if the model has a systematic tendency to over or under-predict.
   - Positive (+): "Optimistic" (Predicting higher gain than reality).
   - Negative (-): "Conservative" (Predicting lower gain than reality).
