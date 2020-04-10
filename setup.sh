# Install torchtext and kaggle
pip install torchtext==0.5.0
pip install -q kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download Flickr30K dataset from Kaggle
!kaggle datasets download -d hsankesara/flickr-image-dataset
!unzip flickr-image-dataset.zip
!rm flickr-image-dataset.zip
!rm -r flickr30k_images/flickr30k_images/flickr30k_images
!rm flickr30k_images/flickr30k_images/results.csv