# Machine-Learning-Application
Several Project examples of ML Applications

#ReinforcementLearning
Participants perform a decision-making task and the their performance will be model using RL models to provide insight on learning and feedback processing.
After model-fitting, regressions is done with EEG data to explore the interaction between neuroscience and behavior.

Start.mlx 
- Runs the RL models on the behavioral data.

6 types of models (These and other tested models can be found under 'ReinforcementLearning/Models')
'StableSimpleQ': Standard Q learning with 2 parameters alpha, beta
'Neg2Q': Q learning with alpha seperated by valence of Prediction Error
'Pun2Q': Q learning with alpha seperated by valence of Reward
'StableSimpleN2Q': Q learning with alpha seperated for Approach vs Avoidance Learning which also includes a single weight
'NegPun2N4Q': Hybrid model with 8 alphas with single weight parameter
'NegPun2N5Q': Hybrid model with 8 alphas and seperate weight parameters

Output of the model with go to '\ReinforcementLearning\SampleData\Plot' as an Excel sheet with following Sheets:
Summary_LLE: Shows model fitness for each model.
Summary_AIC: Shows model fitness corrected for number of parameters
Summary_PseudoR: Shows how much each model explains the behavioral data
Model: Each model will have it's own sheet and shows the fitted free model parameters.

Regression.mlx
- Runs a regression of the EEG data to the output of RL model

Output of RL model are seperated by Valence and Magnitude
EV -> Expected Value
EV_Next -> Expected Value after adjustment
EV_Other -> Expected value of Avoidance
PE -> Prediction error
R -> Reward

#TransferLearningCNN

Image classification has been made easy through models trained on extremely large datasets. These models are able to transfer and applied on other images with very few adjustments. 

With EEG, applying a complex wavelet frequency decomposition allows the creation of "images" with frequency being the y-axis and time being the x-axis. Normally, these images are considered grayscale as only Total Power is used. However, recent studies have shown the importance of phase angle and phase preference in neuro-communication. As such, to further fit EEG data to the RGB transfer model, these two addition metrics are used.

TransferLearning.mlx

-Loads SampleData EEG and runs a Transfer Model.
-This will require downloading the specific transfer models from MATLAB
-Each model will be automatically adjusted to fit the output classification for the EEGlabel and the new adjusted model will be saved.
-Efficientnetb0 was already downloaded and modofied and saved.

#CNN4EEG

Image classification do not transfer exactly onto the EEG domain. While real-world classification are spatially-independent (A cat is still a cat if it's in a different location on the image), EEG data are not (Each frequency-band has signficance at specific times). Additionally, EEG data are not 2D, but 3D with the inclusion of electrode space, which can provide and increase classification when considered together.
As such, a CNN network needs to be created to keep these factors inconsideration. This would involve creating a CNN network framework from scratch.

CNN_Customized.mlx
-Runs a basic CNN network on and example 2D grayscale image dataset with 3 classiciations.
