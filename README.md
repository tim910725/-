# Face Analysis Flask App

## 專案介紹

本專案是一個基於 Flask 架構的人臉分析系統，透過 MediaPipe 進行臉部特徵偵測，並依據五官特徵、膚色類型以及臉型進行時尚建議，包括：

* 臉型辨識（圓形、方形、長形、心形等）
* 膚色判斷（冷色調、暖色調、中性色調）
* 五官分析（眉型、眼型、鼻型）
* 妝容建議（粉底、口紅、眼影、腮紅、鼻影）
* 服裝風格推薦


---

## 安裝步驟

1. Clone 專案到本地：

   ```bash
   git clone <https://github.com/tim910725>
   cd <位置>
   ```

2. 安裝相依套件：

   ```bash
   pip install -r requirements.txt
   ```

3. 執行 Flask 伺服器：

   ```bash
   python index.py
   ```

4. 開啟瀏覽器訪問：
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## API 說明

### 1️⃣ `/detect_face` - 臉部偵測

* **方法**：POST
* **資料格式**：JSON
* **參數**：

  * `image`: Base64 格式的圖片資料
* **回應**：

  ```json
  {
      "face_detected": true
  }
  ```

### 2️⃣ `/analyze` - 臉部分析

* **方法**：POST
* **資料格式**：JSON
* **參數**：

  * `image`: Base64 格式的圖片資料
* **回應**：

  * 臉型、眉型、眼型、鼻型
  * 膚色類型與季節分析
  * 妝容與服裝建議
  * LINE Notify 發送結果

  ```json
  {
      "face_shape": "圓形臉",
      "brow_shape": "細眉",
      "eye_shape": "標準眼",
      "nose_shape": "標準鼻",
      "tone": "暖膚色",
      "season_type": "春季型",
      "color_suggestions": "珊瑚色、鮭粉色、亮金色、淺橘色",
      "style_suggestion": "建議選擇有縱向視覺效果的領型",
      "foundation_suggestion": "選擇偏黃調粉底（如自然色），貼合膚色",
      "nose_shadow_suggestion": "於鼻樑輕掃陰影與高光，自然修飾鼻型",
      "blush_suggestion": "使用珊瑚色或蜜桃色腮紅增添氣色",
      "eyeshadow_suggestion": "適合大地色、古銅色等暖色眼影",
      "eyeliner_suggestion": "沿著眼型畫自然眼線，眼尾略微拉長即可",
      "lipstick_suggestion": "適合珊瑚橘、磚紅等暖色口紅色號",
      "notify_status": "Line Notify 通知已發送"
  }
  ```

---

