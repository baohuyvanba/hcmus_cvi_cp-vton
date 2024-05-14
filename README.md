# Toward Characteristic-Preserving Image-based Virtual Try-on Network

## Dataset:
Mô hình sử dụng bộ dataset VITON-resize được sửa đổi từ bộ dữ liệu VITON để phù hợp với thiết kế mô hình. Dữ liệu có thể tải về theo link [GoogleDrive](https://drive.google.com/open?id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo).
## GMM (Geometric Matching Module)
### Train (quá trình huấn luyện)
Để thực hiện quá trình train của môđun GMM (yêu cầu thiết bị train phải hỗ trợ GPU), ta tiến hành chạy file `run_train_gmm.sh` sau:
``` shell
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  train.py \
    --dataroot='/kaggle/input/vton-cp-resized/viton_resize' \
    --name='GMM' --stage='GMM' --workers=1 --checkpoint_dir='/kaggle/working/' \
    --save_count=10000 --keep_step=50000 --decay_step=50000
```
Trong đó, ta có thể tùy theo cấu hình máy mà thay đổ các tham số `--nnodes=1` số lượng các thiết bị training, `--npro_per_node=1` số lượng các nhân GPU trên thiết bị và khi đó phải thực hiện thay đổi tham số`--workers=1`.

### Evaluation (đánh giá)
Để thực hiện kiểm thử trên tập dữ liệu, ta chạy file `run_test_gmm.sh`
``` shell
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  test.py \
    --dataroot='/kaggle/input/vton-cp-resized/viton_resize' \
    --name='GMM' --stage='GMM' --workers=1 --checkpoint='/kaggle/input/gmm/pytorch/gmmfinal100k/1/gmm_final.pth' \
    --data_list='/kaggle/input/vton-cp-resized/viton_resize/test_pairs.txt' --datamode='test'
```

## Try-on Module (TOM)
### Train (quá trình huấn luyện)
Trước khi thực hiện train Try-on Module, ta sẽ cần sử dụng môđun GMM để tạo ra dữ liệu tạo thư mục `warped-mask` và `warped-cloth` sử dụng lệnh `bash run_gmm_dataset`. Các kết quả `warped-mask` và `warped-cloth` sẽ là đầu vào cho mô hình TOM. Dữ liệu có được sau quá trình thực thi môđun GMM ta kết hợp với bộ dữ liệu ban đầu làm đầu vào cho môđun TOM. Dữ liệu có thể tải về tại [Google Drive](https://drive.google.com/file/d/14vS4Thf7ma3Q4uXdvLnpzhSJ0SLgWLaQ/view?usp=drive_link)<br>
Ta thực thi file `train_train_tom.sh`
``` shell
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  train.py \
    --dataroot='/kaggle/input/vton-cp-resized/viton_resize' \
    --name='TOM' --stage='TOM' --workers=1 --checkpoint_dir='/kaggle/working/' \
    --save_count=10000 --keep_step=50000 --decay_step=50000
```

### Evaluation (đánh giá)
Để thực hiện kiểm thử trên tập dữ liệu, ta chạy file `run_test_gmm.sh`
``` shell
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29400 \
  test.py \
    --dataroot='/kaggle/input/vton-cp-resized/viton_resize' \
    --name='TOM' --stage='TOM' --workers=1 --checkpoint='/kaggle/input/tom/pytorch/tomfinal100k/1/tom_final.pth' \
    --data_list='/kaggle/input/vton-cp-resized/viton_resize/test_pairs.txt' --datamode='test'
```

## References
Mô hình trên được cài đặt dựa trên bài viết:

@inproceedings{wang2018toward,
	title={Toward Characteristic-Preserving Image-based Virtual Try-On Network},
	author={Wang, Bochao and Zheng, Huabin and Liang, Xiaodan and Chen, Yimin and Lin, Liang},
	booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	pages={589--604},
	year={2018}
}
