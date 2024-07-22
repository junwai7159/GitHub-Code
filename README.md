## About this project
Respiratory rate waveforms play a vital role in detecting to sleep disorders, such as sleep apnea hypoventilation syndrome require long-term continuous respiratory monitoring through hospital or sleep center. Portable apnea testing sensors can help patients who find it difficult to attend appointments in these environments, as well as providing a more comfortable, non-contact experience. Sleep staging detects the quality of sleep as well as the health status of people, and traditional manual sleep staging is rather cumbersome and time-consuming. Automatic sleep staging as multi-classification problem, the application of deep learning to sleep staging can realize the automation of sleep staging. In this paper, we propose a non-contact respiration detection algorithm based on microphone arrays, based on ultrasonic perception which utilizes FMCW to convert the measured human chest movement frequency into human respiration frequency. This respiration frequency is then used as a feature input to an automatic sleep staging model based on CNN and LSTM to automatically classify sleep stages during sleep to achieve non-contact respiration detection as well as sleep staging. 

## Features
The code in this repo only implements the automatic sleep staging model, a deep learning based hybrid CNN-BiLSTM network. 
- CNN to extract features related to sleep stages from the input sample sequences
- BiLSTM for capturing the forward & backward temporal dependencies within the sequences
- The feature space of input data consists of 4 features (RR Internal Average, RR Internal Variance, RR Internal Difference Average, Respiratory Rate)
- 4 types of sleep stages based on AASM: Wake, Light (N1/N2), Deep (N3), REM

The model architecture is shown below:

<img src="https://github.com/user-attachments/assets/5df06b68-24b6-4ef2-97dd-748070da4ec7" width="800">

## Usage
Run `sleep_rnn.ipynb` for data preprocessing, training, and testing the model

## Demonstrations
#### Test Subject 16
| Hypnogram | Confusion Matrix | 
| --- | --- |
| ![image](https://github.com/user-attachments/assets/ad592589-cbe5-4700-a04d-adeab6d67d1d) | ![image](https://github.com/user-attachments/assets/5969fb1c-8a21-4d47-b9f0-017b29e6d859) |

#### Test Subject 17
| Hypnogram | Confusion Matrix | 
| --- | --- |
| ![image](https://github.com/user-attachments/assets/f4edd030-199d-4edd-bb3e-fa44da5940bc) | ![image](https://github.com/user-attachments/assets/79860c85-4298-4c40-b901-6f2f1118c0a4) |

