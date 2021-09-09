Suggestion env:
python3.7
pytorch==1.2
torchvision==0.4

------------
prepare sprite dataset based on https://github.com/YingzhenLi/Sprites
put the files in "npy" into dataset/Sprite

train_DS_VAE_sprite.py:  train the S3VAE model
video_classifier_Sprite_all.py: train the classifier for testing 
test_DS_VAE_Sprite_Cls_disagree.py:  reproduce the scores 
saved_model is for testing

You can run:
python3 test_DS_VAE_Sprite_Cls_disagree.py --type_gt action
python3 test_DS_VAE_Sprite_Cls_disagree.py --type_gt skin
python3 test_DS_VAE_Sprite_Cls_disagree.py --type_gt top
python3 test_DS_VAE_Sprite_Cls_disagree.py --type_gt pant
python3 test_DS_VAE_Sprite_Cls_disagree.py --type_gt hair
