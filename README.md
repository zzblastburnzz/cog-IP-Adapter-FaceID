# Ancient Portrait Generator

Tạo ảnh chân dung cổ trang phong cách tiên hiệp từ ảnh khuôn mặt đầu vào.

## Inputs
- `face_image`: Ảnh khuôn mặt đầu vào
- `vip_level`: Cấp VIP (Đồng, Bạc, Vàng, Bạch Kim, Kim Cương)
- `profession`: Nghề nghiệp (Kiếm Khách, Đao Khách, Pháp Sư, Y Sư, Cung Thủ, Tu Sĩ)
- `gender`: Giới tính (Nam, Nữ)
- `seed`: Random seed (optional)

## Output
- Ảnh chân dung cổ trang với kích thước 9:16, phong cách tiên hiệp

## Example
```python
output = predict(
    face_image="path/to/face.jpg",
    vip_level="Kim Cương",
    profession="Pháp Sư",
    gender="Nữ"
)