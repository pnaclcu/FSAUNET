# Pytorch Implement for Full-spectrum attention U-Net 
## Repo architecture
unet/unet_parts.py and unet_model.py include the U-Net model.
unet/full_freq_att.py includes the proposed full-spectrum model.
utils/dataset.py is used to process dataloader.
dice_loss.py is used to compute the dice scores.

plz see the details in the following introduction.

## Dataset preparation

The CAMUS echocardiography dataset is recommended !! 

If you do not want to rewrite the utils/dataset.py, please using the following dataset structure.

- CAMUS |  
  - training >  
    - patinet0001 >
		- img >
			- img0001.png, img0002.png.....
		- mask >
			- img0001.png, img0002.png.....
 - testing >  
	As same as the training.
			
The name of image and its correspongding mask should be same. 

## Training the model
run python train.py
## Predicting the model
run python predict.py
