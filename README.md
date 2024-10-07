LT2326/LT2926 
Assignment 1   - Thai & English OCR
name: Daniilidou Viktoria Paraskevi

How to run the scripts:
1.	Splitting the dataset & challenges
In order to get the splitting of the data you have to run on the server this command :
 
python splitdata.py 'Thai_English_normal' 'All_Thai_styles' 'Thai_bold_400dpi' /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet

in this path 'Thai_English_normal' is the training, 'All_Thai_styles' is the validation and 'Thai_bold_400dpi' is the test.
 
Splitting the data was challenging for two reasons. The first reason was to be able to separate and call the bold italic from the bold and the second one was the difficulty to ignore and take only one language, 
when I wanted for example just the Thai bold 200dpi category.  
Also, when I was working on this assignment in the beginning I formed my code in a jupyter notebook so in order not to change it for the script the train/validation/test split is hardcoded in the split_train_val_test() function, 
something that I know it’s not efficient and  It would be more beneficial to allow language style and dpi to be configurable from the command line. So, the datasets you can choose for training test and validation are those based on the board that was on canvas. 
'Thai_normal_alldpi', 'Thai_normal_200dpi', 'Thai_normal_400dpi', 'Thai_bold_alldpi', 'All_Thai_styles', 'Thai_English_normal', 'Thai_bold_400dpi', 'Thai_bold_italic', 'All_Thai_English_styles'. 

2.	Creating the model & challenges
In order to get the training you have to run on the server this command :

for the training: 
python train.py --epochs 10 --batch_size 32

and for the test:
python test.py --epochs 10 --batch_size 32

model.py
I chose in my model to have three layer CNN with increasing filter sizes in order to capture more complex features of the images and I used MaxPooling after each layer so that the spatial dimensions can be reduced. 
I also used x.view(x.size(0), -1) to flatten the tensor for the fully connected layers, so I can get from the multidimensional convolutional layers, a 1-dimensional vector. I added a Dropout layer (0.5) to randomly drop neurons during training, 
helping the model generalize better and batch normalization to make training more stable. 

dataset.py
In this script I have the extract_unique_labels() function in order to extract and sort the unique labels from the image paths. 
Also, inside the ThaiEngOCRDataset Class only the .bmp paths are stored and mapped to indices and returns the transformed image and its corresponding label. 

train.py
Training the model: In order to ensure that the images have the same format, I used transforms in order to resize the images to 32x32. 
I created  a mapping from labels to indices and NLLLoss as a loss function. Before each batch I used  optimizer.zero_grad() to clear gradients and I compute loss and update the model using backpropagation. 

test.py
Evaluates the model using the sklearn metrics accuracy, precision, recall and F1-score on unseen data.

validation.py
Evaluates the Thai OCR model using the sklearn metrics accuracy, precision, recall and F1-score during training.


Checking all the possible character resolutions from the board by running the training  we can observe that the loss is getting lower so the model is learning and improving its performance. 
Also the accuracy, precision, recall and F1-score are high which indicates that the model performs well. 

I also uploaded a pdf report with screenshots with all the experiments of training and testing as here I cannot add images. 
