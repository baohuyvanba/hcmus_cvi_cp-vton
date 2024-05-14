# Toward Characteristic-Preserving Image-based Virtual Try-on Network

## Dataset:
Ta lấy dataset VITON theo link [GoogleDrive](https://drive.google.com/open?id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo).


## GMM (Geometric Matching Module)
### Train (huấn luyện)
Ta có thể chạy lệnh train như sau. Ngoài ra có các tham số khác có thể tinh chỉnh cho phù hợp. 
```
python train.py --name gmm_train_new --stage GMM --workers 4 --save_count 5000 --shuffle
```

### Eval (đánh giá)
Chuyển data sang chế độ test. Một ví dụ cho lệnh test:
```
python test.py --name gmm_traintest_new --stage GMM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/gmm_train_new/gmm_final.pth
```

## Try-on Module (TOM)
### Train
Trước khi thực hiện train Try-on Module, ta sẽ tạo thư mục warped-mask và warped-cloth là kết quả của hàm train GMM trước đó. Các kết quả warped-mask và warped-cloth sẽ là đầu vào cho mô hình TOM. Một ví dụ cho lệnh train TOM (đổi "--stage" sang "TOM"):

```
python train.py --name tom_train_new --stage TOM --workers 4 --save_count 5000 --shuffle 
```
### Eval
Tương tự như bước train, ta cũng cần tạo warped-cloth và warped-mask tương ứng với tập dữ liệu test. Ví dụ cho lệnh chạy test:
```
python test.py --name tom_test_new --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/tom_train_new/tom_final.pth
```

## Citation
If this code helps your research, please cite our paper:

	@inproceedings{wang2018toward,
		title={Toward Characteristic-Preserving Image-based Virtual Try-On Network},
		author={Wang, Bochao and Zheng, Huabin and Liang, Xiaodan and Chen, Yimin and Lin, Liang},
		booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
		pages={589--604},
		year={2018}
	}
