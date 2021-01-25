# MedicalImageSegmentation
Take home exam done for a medical imaging company. Images were originally provided as 3D DICOM files, and as I have a 2GB MX150, these had to be converted into 2D slides for segementation

Question 1 was regular segmentation, which was performed with a modified UNet. Results for Dice and Jaccard score were 0.97 and 0.94 on the validation set, respectively.
![Sample Unet Images](https://github.com/JonathanBhimani-Burrows/MedicalImageSegmentation/Unet_Images.PNG)

Question 2 involved implementing pix2pix, which is a combined segmentation/GAN network. The generator is the Unet from part 1, and the discriminator is the encoder of the Unet. While results for this approach aren't great - due to a lack of computational resources - it is clear that the network is working and that the GAN was slowly getting results.
![Sample GAN Images](https://github.com/JonathanBhimani-Burrows/MedicalImageSegmentation/GAN_Images.PNG)
