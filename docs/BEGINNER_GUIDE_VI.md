# Đọc và hiểu cách chạy DetZero từ đầu đến cuối

Tài liệu này dành cho người mới biết Python cơ bản. Các lệnh là hướng dẫn cho máy Linux có GPU NVIDIA và môi trường phù hợp; chúng chưa được chạy để kiểm chứng trong phiên này. Máy Mac hiện tại dùng ARM, và bản checkout hiện chỉ có `data/waymo/ImageSets`, chưa có dữ liệu LiDAR hay checkpoint.

## 1. Chương trình nhận gì, trả gì?

Bạn đưa vào chuỗi dữ liệu LiDAR. Mỗi frame là một lần quét, chứa nhiều điểm trong không gian. Chương trình trả về các hộp 3D bao quanh xe, người đi bộ và người đi xe đạp ở từng frame.

Ví dụ mô phỏng bằng Python, không phải định dạng đầy đủ của repo:

```python
frames = [
    {"frame_id": 0, "points": [[1.0, 2.0, 0.5], [1.1, 2.0, 0.6]]},
    {"frame_id": 1, "points": [[1.5, 2.0, 0.5], [1.6, 2.0, 0.6]]},
]
```

Một hộp có bảy giá trị cơ bản:

```python
box = [10.0, 2.0, 1.0, 4.5, 1.8, 1.6, 0.0]
#      x     y    z    dx   dy   dz   heading
#      tâm hộp         kích thước     góc hướng
```

Một frame có thể có nhiều hộp. Một sequence là chuỗi nhiều frame liên tiếp. Một track là các hộp thuộc cùng một vật thể qua nhiều frame.

## 2. Luồng có thể hình dung bằng Python

```python
# Chỉ minh họa ý tưởng, không phải API thật của repo.
frames = preprocess(raw_waymo_data)
detections = detect(frames, detector_checkpoint)
tracks = track(detections)
object_data = crop_points_for_each_object(frames, tracks)

geometry = refine_size(object_data, geometry_checkpoint)
position = refine_position_and_heading(object_data, position_checkpoint)
confidence = refine_score(object_data, confidence_checkpoint)

final_result = combine(geometry, position, confidence)
evaluate(final_result, ground_truth)
```

Trong project thật, mỗi bước thường là một chương trình Python riêng. Chương trình trước lưu kết quả ra file `.pkl`; chương trình sau đọc file đó.

```text
Dữ liệu Waymo thô
    → tiền xử lý: point cloud .npy và thông tin .pkl
    → detection: hộp theo frame
    → tracking: hộp theo vật thể qua thời gian
    → chuẩn bị object data: cắt điểm LiDAR cho từng vật thể
    → GRM, PRM, CRM: ba nhóm kết quả tinh chỉnh
    → ghép: hộp và score cuối cùng theo frame
    → đánh giá bằng nhãn đúng
```

Ví dụ: một chiếc xe được nhìn thấy ở nhiều frame. Detector dự đoán kích thước hơi khác nhau giữa các frame. Tracker xác định đó là cùng một xe. GRM dùng nhiều quan sát để sửa kích thước; PRM sửa tâm và hướng ở từng thời điểm; CRM sửa độ tin cậy của các dự đoán.

## 3. Các từ cần biết trước khi chạy

| Từ | Ý nghĩa trong bài này |
| --- | --- |
| Model | Bộ xử lý có các tham số học được từ dữ liệu |
| Train | Dùng dữ liệu có nhãn để điều chỉnh các tham số |
| Inference | Dùng model đã học để dự đoán trên dữ liệu |
| Checkpoint `.pth` | File lưu trạng thái model đã huấn luyện |
| Config `.yaml` | File cài đặt loại model, dữ liệu, batch size... |
| Ground truth / GT | Nhãn đúng để huấn luyện hoặc chấm kết quả |
| Batch | Một nhóm mẫu được xử lý cùng lúc |
| Epoch | Một lượt huấn luyện qua tập train, theo cấu hình |
| `.npy` | File lưu mảng NumPy, ở đây có thể là point cloud |
| `.pkl` | File lưu đối tượng Python như list, dict, mảng |
| `train` / `val` / `test` | Các phần dữ liệu để học, kiểm tra và dự đoán cuối |

`test.py` là tên script chạy dự đoán/đánh giá. Mặc định config của repo ánh xạ chế độ `test` sang tập `val`, nên chạy `test.py` không có nghĩa là luôn dùng tập Waymo test.

## 4. Đọc một câu lệnh

```bash
python test.py \
  --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml \
  --ckpt /path/to/detector.pth \
  --batch_size 1
```

- `python test.py`: chạy file `test.py` trong thư mục hiện tại.
- `--cfg_file ...`: chọn bản cấu hình model và dữ liệu.
- `--ckpt ...`: nạp tham số đã học từ file checkpoint.
- `--batch_size 1`: xử lý mỗi batch một mẫu.
- Dấu `\` nối một câu lệnh dài qua nhiều dòng.

Code dùng `argparse` để chuyển các tham số này thành thuộc tính Python như `args.cfg_file` và `args.ckpt`.

Nhiều đường dẫn config là tương đối với thư mục hiện tại. Vì vậy, cần `cd` đúng thư mục trước khi chạy.

## 5. Điều kiện để chạy bản thật

Cần có:

1. Máy Linux có GPU NVIDIA, driver và CUDA phù hợp.
2. Môi trường Python cùng các dependency đã cài và biên dịch được.
3. Dữ liệu Waymo; các file `ImageSets/*.txt` chỉ liệt kê tên sequence, không chứa point cloud.
4. Checkpoint của detector và các model refining đúng với config.

Môi trường tác giả ghi trong `docs/INSTALL.md` gồm Python 3.8, PyTorch 1.10+, CUDA 11+ và spconv v2. Đây là thông tin của bản repo, không phải cam kết rằng các phiên bản package mới nhất có thể cài chung. Khi triển khai thật cần xác nhận bộ phiên bản trên máy GPU.

Sau khi cài dependency, repo hướng dẫn đăng ký/biên dịch các package như sau, chạy từ thư mục gốc DetZero:

```bash
(cd utils && python setup.py develop)
(cd detection && python setup.py develop)
(cd tracking && python setup.py develop)
(cd refining && python setup.py develop)
```

Dấu ngoặc tạo shell con: `cd` bên trong không làm đổi thư mục của shell bên ngoài. Bước này giúp Python import được `detzero_utils`, `detzero_det`, `detzero_track`, `detzero_refine`; phần `utils` còn biên dịch các phép toán CUDA.

Các bước bên dưới giả định môi trường đã được chuẩn bị thành công.

## 6. Dùng model có sẵn: luồng end to end trên tập val

Mục tiêu là hiểu và chạy pipeline dự đoán; bạn chưa cần huấn luyện model.

### Bước A: chuẩn bị đường dẫn và dữ liệu

Trong terminal của máy Linux giả định:

```bash
export DETZERO_ROOT=/path/to/DetZero
export DETZERO_CKPT_DIR=/path/to/checkpoints
```

Thay hai đường dẫn trên bằng đường dẫn thật. `$DETZERO_ROOT` và `$DETZERO_CKPT_DIR` là các biến shell để tránh gõ lại.

Cấu trúc dữ liệu:

```text
DetZero/data/waymo/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── raw_data/
    └── segment-....tfrecord
```

Các sequence được liệt kê trong ImageSets phải tương ứng với dữ liệu bạn có. Script tiền xử lý hiện duyệt cả train, val, test; nó không cung cấp tham số chỉ chọn val.

Để thống nhất ví dụ, giả sử bạn đã đặt checkpoint theo tên sau. Đây là quy ước minh họa, không phải tên file có sẵn trong repo:

```text
checkpoints/
├── detector.pth              # phù hợp centerpoint_1sweep.yaml
├── vehicle_grm.pth
├── vehicle_prm.pth
├── vehicle_crm.pth
├── pedestrian_grm.pth
├── pedestrian_prm.pth
├── pedestrian_crm.pth
├── cyclist_grm.pth
├── cyclist_prm.pth
└── cyclist_crm.pth
```

Các config refining mặc định dùng model riêng cho từng loại vật thể. Để chạy đủ ba loại, ví dụ này cần chín checkpoint refining. Đổi tên file không làm checkpoint tương thích với một kiến trúc khác.

### Bước B: tiền xử lý Waymo

```bash
cd "$DETZERO_ROOT/detection"
python -m detzero_det.datasets.waymo.waymo_preprocess \
  --cfg_file tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml \
  --func create_waymo_infos
```

`python -m ...` yêu cầu Python chạy một module theo đường dẫn package.

Đầu vào: file `.tfrecord`. Đầu ra chính:

```text
data/waymo/
├── waymo_infos_train.pkl
├── waymo_infos_val.pkl
├── waymo_infos_test.pkl
└── waymo_processed_data/
    └── segment-.../
        ├── 0000.npy
        ├── 0001.npy
        └── segment-....pkl
```

Các file `.npy` chứa điểm của frame; các file thông tin mô tả đường dẫn, nhãn và pose. Pose là ma trận giúp chuyển tọa độ giữa hệ LiDAR và hệ tọa độ chung.

Lệnh trong tài liệu gốc có chỗ dùng `waymo_one_sweep.yaml`; tên file thực tế trong checkout này là `waymo_1sweep.yaml`.

### Bước C: detection

```bash
cd "$DETZERO_ROOT/detection/tools"
python test.py \
  --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml \
  --ckpt "$DETZERO_CKPT_DIR/detector.pth" \
  --local \
  --batch_size 1 \
  --workers 1 \
  --extra_tag e2e
```

Đầu vào: dữ liệu đã xử lý và checkpoint detector. Đầu ra: `result.pkl`, một list chứa kết quả theo frame.

Luồng code:

```text
test.py: main()
    → đọc config
    → tạo dataset và dataloader
    → tạo CenterPoint và nạp checkpoint
    → eval_utils.eval_one_epoch()
        → lấy từng batch
        → model(batch_dict)
        → chuyển dự đoán thành kết quả theo frame
        → lưu result.pkl và đánh giá
```

Đường dẫn kết quả có dạng:

```text
detection/output/det_model_cfgs/centerpoint_1sweep/e2e/
    eval/epoch_.../val/result.pkl
```

Tên `epoch_...` được script suy ra từ đường dẫn checkpoint. Dùng đường dẫn được in trong log, không tự đoán số epoch.

Gán đường dẫn thật đó để dùng ở bước sau:

```bash
export DETZERO_DET_RESULT=/absolute/path/to/result.pkl
```

### Bước D: tracking

```bash
cd "$DETZERO_ROOT/tracking/tools"
python run_track.py \
  --cfg_file cfgs/tk_model_cfgs/waymo_detzero_track.yaml \
  --data_path "$DETZERO_DET_RESULT" \
  --split val \
  --batch_size 1 \
  --workers 1
```

Đầu vào: detection theo frame và thông tin GT của val. Đầu ra nằm trong `data/waymo/tracking/`, gồm file tracking và file `drop` lưu các detection bị tách ra khi lọc chồng lấn.

Tracking ghép các hộp có khả năng thuộc cùng vật thể. Với `--split val`, code cũng ghép track với nhãn đúng để phục vụ đo chất lượng và chuẩn bị refining; việc này không huấn luyện trọng số tracker.

Gán đúng hai đường dẫn tạo ra ở lần chạy này:

```bash
export DETZERO_TRACK_RESULT=/absolute/path/to/tracking-val-TIMESTAMP.pkl
export DETZERO_DROP_RESULT=/absolute/path/to/drop-val-TIMESTAMP.pkl
```

### Bước E: cắt điểm cho từng vật thể

```bash
cd "$DETZERO_ROOT/daemon"
python prepare_object_data.py \
  --track_data_path "$DETZERO_TRACK_RESULT" \
  --split val \
  --workers 1
```

Đầu vào: tracking cùng point cloud đã tiền xử lý. Đầu ra: dữ liệu từng vật thể trong các thư mục:

```text
data/waymo/refining/Vehicle/
data/waymo/refining/Pedestrian/
data/waymo/refining/Cyclist/
```

Mỗi file sequence chứa các vật thể, hộp theo thời gian và các điểm LiDAR quanh chúng. Nếu chỉ có hộp tracking mà không có point cloud gốc, bước này chưa chạy được.

### Bước F: chạy ba loại refining cho ba loại vật thể

```bash
cd "$DETZERO_ROOT/refining/tools"
for object_class in vehicle pedestrian cyclist; do
  for refine_kind in grm prm crm; do
    python test.py \
      --cfg_file "cfgs/ref_model_cfgs/${object_class}_${refine_kind}_model.yaml" \
      --ckpt "$DETZERO_CKPT_DIR/${object_class}_${refine_kind}.pth" \
      --batch_size 1 \
      --workers 1 \
      --extra_tag e2e \
      --save_to_file
  done
done
```

Hai vòng lặp shell trên chạy tổng cộng chín lệnh. Ví dụ ở lượt `vehicle` và `grm`, chúng chọn `vehicle_grm_model.yaml` và `vehicle_grm.pth`.

- GRM tạo kết quả kích thước.
- PRM tạo kết quả vị trí và hướng.
- CRM tạo score mới.
- `--save_to_file` cần có để lưu dự đoán refining cho bước ghép.
- Batch size 1 giúp minh họa cách chạy với ít mẫu mỗi lượt; mức dùng RAM/VRAM vẫn phụ thuộc dữ liệu và model.

Các file đầu ra có tên như:

```text
Vehicle_geometry_val.pkl
Vehicle_position_val.pkl
Vehicle_confidence_val.pkl
```

Tương tự với `Pedestrian` và `Cyclist`. Chúng nằm trong các thư mục `refining/output/ref_model_cfgs/.../e2e/eval/epoch_.../val/` tương ứng.

### Bước G: tập hợp và ghép kết quả

Script ghép tìm đầu vào trong **`data/waymo/refining/result/`**. Chú ý tên là `result` số ít.

```bash
mkdir -p "$DETZERO_ROOT/data/waymo/refining/result"
```

Copy chín file từ bước F vào đó. Ví dụ một file, thay đường dẫn bên trái bằng đường dẫn thật trong log:

```bash
cp /absolute/path/to/Vehicle_geometry_val.pkl \
  "$DETZERO_ROOT/data/waymo/refining/result/"
```

Khi đã có đủ chín file:

```bash
cd "$DETZERO_ROOT/daemon"
python combine_output.py \
  --split val \
  --combine_conf_res \
  --combine_drop_path "$DETZERO_DROP_RESULT" \
  --track_save \
  --frame_save
```

Chương trình lấy tâm/hướng từ PRM, kích thước từ GRM, score từ CRM. Nó xuất kết quả theo track và theo frame. Việc thêm lại detection trong file drop là lựa chọn của lệnh này và cũng là cách tài liệu gốc minh họa đánh giá.

Một chi tiết của bản code hiện tại: `combine_dict` tích lũy qua vòng lặp các lớp. Với cùng tập sequence, file ở lượt cuối `Cyclist_final_frame.pkl` chứa kết quả tích lũy các lớp, dù tên file dễ gây hiểu nhầm. Khi chạy thật hãy kiểm tra trường `name` trong kết quả và log để xác nhận các lớp đã được ghép đủ.

### Bước H: đánh giá

```bash
cd "$DETZERO_ROOT/evaluator"
python detzero_eval.py \
  --det_result_path "$DETZERO_ROOT/data/waymo/refining/result/Cyclist_final_frame.pkl" \
  --gt_info_path "$DETZERO_ROOT/data/waymo/waymo_infos_val.pkl" \
  --evaluate_metrics object
```

Đầu vào là dự đoán cuối và nhãn đúng. Chương trình tính các chỉ số đánh giá Waymo và lưu log `mAP.txt` cạnh file kết quả. Có file đầu ra nghĩa là pipeline đã tạo được dự đoán; chỉ số đánh giá mới cho biết dự đoán tốt đến đâu.

## 7. Nếu chưa có checkpoint thì huấn luyện theo thứ tự nào?

Đây là lộ trình bổ sung, khác với việc chạy dự đoán bằng checkpoint có sẵn:

```text
Tiền xử lý dữ liệu
    → tạo database augmentation cho detector
    → train detector
    → detector dự đoán trên train và val
    → tracking trên train và val
    → chuẩn bị object data cho train và val
    → train GRM và PRM
    → chạy GRM và PRM trên train
    → tính nhãn IoU cho CRM
    → train CRM
    → chạy ba model trên val
    → ghép kết quả và đánh giá
```

Các lệnh mẫu cho nhánh Vehicle, sau khi đã chuẩn bị đủ đầu vào từng bước:

```bash
# Database các vật thể thật để tăng cường dữ liệu khi train detector.
cd "$DETZERO_ROOT/detection"
python -m detzero_det.datasets.waymo.waymo_preprocess \
  --cfg_file tools/cfgs/det_dataset_cfgs/waymo_1sweep.yaml \
  --func create_waymo_database

cd "$DETZERO_ROOT/detection/tools"
python train.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml --local

# Sau detection, tracking và chuẩn bị object data cho cả train và val:
cd "$DETZERO_ROOT/refining/tools"
python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml
python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_prm_model.yaml
```

Để detector dự đoán tập train, dùng `test.py` với checkpoint detector và thêm `--set DATA_CONFIG.DATA_SPLIT.test train`. Tracking và chuẩn bị object data tương ứng dùng `--split train`.

Sau khi GRM/PRM đã học, chạy hai model trên train:

```bash
cd "$DETZERO_ROOT/refining/tools"
python test.py \
  --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml \
  --ckpt /absolute/path/to/trained_grm.pth \
  --save_to_file \
  --set DATA_CONFIG.DATA_SPLIT.test train

python test.py \
  --cfg_file cfgs/ref_model_cfgs/vehicle_prm_model.yaml \
  --ckpt /absolute/path/to/trained_prm.pth \
  --save_to_file \
  --set DATA_CONFIG.DATA_SPLIT.test train

cd "$DETZERO_ROOT/daemon"
python generate_iou_gt.py \
  --class_name Vehicle \
  --geo_path /absolute/path/to/Vehicle_geometry_train.pkl \
  --pos_path /absolute/path/to/Vehicle_position_train.pkl

cd "$DETZERO_ROOT/refining/tools"
python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_crm_model.yaml
```

IoU đo độ chồng khớp giữa hộp dự đoán và hộp đúng. Bước tạo nhãn IoU cho biết chất lượng hộp sau GRM/PRM để CRM học cách đánh giá độ tin cậy. Lặp quy trình refining tương ứng cho Pedestrian và Cyclist nếu cần đủ ba lớp.

README refining có một số lệnh PRM trỏ nhầm sang file GRM. PRM cần config `*_prm_model.yaml`.

## 8. Nối luồng chạy với kiến thức Python cơ bản

Ví dụ Python thông thường:

```python
def main():
    config = read_config()
    data = read_data(config)
    processor = create_processor(config)
    result = processor(data)
    save_result(result)

if __name__ == "__main__":
    main()
```

Các script DetZero cũng có khung điều phối tương tự, nhưng phần đọc dữ liệu và model nằm trong package khác.

Trong PyTorch, `model(batch_dict)` sẽ dẫn đến `forward(batch_dict)` qua cơ chế gọi của `nn.Module`. Bạn không cần tìm một hàm tên `__call__` do tác giả tự viết trong CenterPoint.

`batch_dict` chỉ là dictionary chứa điểm, voxel, đặc trưng, nhãn... Các module đọc và thêm key:

```python
# Mô phỏng cách truyền dữ liệu, không phải code chạy thật.
batch_dict = {"voxels": voxel_array}
batch_dict["voxel_features"] = encode_voxels(batch_dict["voxels"])
batch_dict["spatial_features"] = extract_features(batch_dict["voxel_features"])
```

Khi đọc một hàm, hãy ghi bốn điều: ai gọi nó, nó đọc key/file nào, nó xử lý gì, và nó tạo key/file nào cho bước sau.

Ba file nên đọc trước:

1. `detection/tools/test.py`: hàm `main()` để thấy chương trình được lắp ráp thế nào.
2. `detection/tools/eval_utils.py`: vòng `for ... in dataloader` để thấy dữ liệu thực sự được xử lý thế nào.
3. `detection/detzero_det/models/centerpoint.py`: hàm `forward()` để thấy các bước xử lý bên trong model.

Sau khi hiểu ba file này, đi ngược về dataset để xem dữ liệu được tạo ra thế nào, rồi đi tiếp tracking và refining.
