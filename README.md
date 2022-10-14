# STCA: Utilizing a spatio-temporal cross-attention network for enhancing video person re-identification

# Credits
The source code is built upon the github repository [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID). Code is developed based on PyTorch Framework.

We would like to thank [jiyanggao] (https://github.com/jiyanggao/Video-Person-ReID) for his generous contribution to release the code to the community.

#Datasets
Dataset preparation instructions can be found in the repository [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID). 

#Testing 

You can run  `sh test.sh ` file or you can run the following command:

` python main_test_32.py  -d mars  --pretrained-model 'Path-to-trained-model/checkpoint_ep450.pth.tar'  --save-dir github `


Trained model for Mars dataset can be download from the [Google Drive](https://drive.google.com/file/d/1v8Ho4K9a8nKAEiHry_61D7t_pzh8A7Hx/view?usp=sharing)


#Citation

```
@article{bhuiyan2022stca,
  title={STCA: Utilizing a spatio-temporal cross-attention network for enhancing video person re-identification},
  author={Bhuiyan, Amran and Huang, Jimmy Xiangji},
  journal={Image and Vision Computing},
  volume={123},
  pages={104474},
  year={2022},
  publisher={Elsevier}
}

```
