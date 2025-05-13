from flask import Flask, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import math
import requests
import base64

app = Flask(__name__, static_folder='.')  # 設置靜態檔案目錄為當前目錄

# 初始化 MediaPipe FaceMesh 模組
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 定義臉部特徵索引
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# 全局變數（移除與 Tkinter 和 ESP32 相關的部分）
lipstick_color = [130, 100, 150]  # 初始口紅顏色 (R, G, B)
suggested_lipsticks = []  # 存放膚色建議的口紅色號
opacity_slider = 50  # 口紅不透明度 (0-100)
gloss_intensity = 0.7  # 高光強度 (0-1)

# 基於膚色與季節類型的建議口紅色號
lipstick_recommendations = {
    "冷膚色": {
        "夏季型": [
            {"name": "柔和玫紅", "color": [180, 120, 160]},
            {"name": "粉紫紅", "color": [190, 130, 180]},
            {"name": "霧面莓果", "color": [150, 80, 120]},
            {"name": "薰衣草紫紅", "color": [160, 100, 170]}
        ],
        "冬季型": [
            {"name": "正紅色", "color": [220, 50, 80]},
            {"name": "紫紅色", "color": [160, 60, 130]},
            {"name": "酒紅色", "color": [140, 40, 80]},
            {"name": "深莓果色", "color": [120, 40, 90]}
        ]
    },
    "暖膚色": {
        "春季型": [
            {"name": "珊瑚橙", "color": [240, 130, 100]},
            {"name": "蜜桃粉", "color": [235, 150, 155]},
            {"name": "溫暖粉紅", "color": [220, 140, 150]},
            {"name": "杏橙色", "color": [230, 140, 120]}
        ],
        "秋季型": [
            {"name": "磚紅色", "color": [180, 80, 70]},
            {"name": "赤褐色", "color": [170, 90, 80]},
            {"name": "南瓜色", "color": [200, 100, 80]},
            {"name": "楓葉紅", "color": [160, 70, 60]}
        ]
    },
    "中性膚色": {
        "通用": [
            {"name": "經典紅", "color": [200, 50, 50]},
            {"name": "豆沙色", "color": [170, 110, 100]},
            {"name": "裸粉色", "color": [220, 150, 140]},
            {"name": "玫瑰棕", "color": [160, 100, 100]}
        ]
    }
}

# 更新膚色建議的口紅色號
def update_lipstick_suggestions(skin_tone, season_type):
    global suggested_lipsticks
    suggested_lipsticks = []
    
    if skin_tone in lipstick_recommendations:
        if skin_tone == "中性膚色":
            suggested_lipsticks = lipstick_recommendations[skin_tone]["通用"]
        elif season_type in lipstick_recommendations[skin_tone]:
            suggested_lipsticks = lipstick_recommendations[skin_tone][season_type]
        else:
            first_season = list(lipstick_recommendations[skin_tone].keys())[0]
            suggested_lipsticks = lipstick_recommendations[skin_tone][first_season]
    else:
        suggested_lipsticks = lipstick_recommendations["中性膚色"]["通用"]
    
    return suggested_lipsticks

# 處理 Base64 影像
def decode_image(image_data):
    img_bytes = base64.b64decode(image_data)
    nparray = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparray, cv2.IMREAD_COLOR)

# 添加口紅光澤效果
def add_lip_gloss(overlay, lip_points, frame_shape, base_color):
    lip_center_x = int(np.mean([p[0] for p in lip_points]))
    lip_center_y = int(np.mean([p[1] for p in lip_points]))
    
    gloss_size_x = int((max([p[0] for p in lip_points]) - min([p[0] for p in lip_points])) * 0.3)
    gloss_size_y = int((max([p[1] for p in lip_points]) - min([p[1] for p in lip_points])) * 0.2)
    
    gloss_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.ellipse(gloss_mask, 
                (lip_center_x, lip_center_y - int(gloss_size_y * 0.3)),
                (gloss_size_x, gloss_size_y),
                0, 0, 360, 255, -1)
    
    gloss_mask = cv2.GaussianBlur(gloss_mask, (21, 21), 0)
    
    gloss_color = [min(int(c * 1.5), 255) for c in base_color[::-1]]  # BGR 格式
    
    gloss_layer = overlay.copy()
    cv2.fillPoly(gloss_layer, [lip_points], gloss_color)
    gloss_alpha = gloss_mask / 255.0 * gloss_intensity
    for c in range(3):
        overlay[:, :, c] = (1 - gloss_alpha) * overlay[:, :, c] + gloss_alpha * gloss_layer[:, :, c]
    
    return overlay

# 創建嘴唇 Alpha Mask
def create_lip_alpha_mask(lip_points, frame_shape, blur_size=15):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [lip_points], 255)
    alpha_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    alpha_mask = alpha_mask / 255.0
    return alpha_mask

# 自動白平衡
def auto_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# 低光增強
def enhance_low_light(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    if brightness < 50:
        return cv2.convertScaleAbs(img, alpha=1.5, beta=20)
    return img

# 主頁路由
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# 人臉偵測路由
@app.route('/detect_face', methods=['POST'])
def detect_face():
    data = request.json
    image_data = data['image'].split(',')[1]
    img = decode_image(image_data)

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    return jsonify({'face_detected': bool(results.multi_face_landmarks)})

# 分析路由
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data['image'].split(',')[1]
    img = decode_image(image_data)

    # 影像預處理
    frame_enhanced = enhance_low_light(img)
    frame_corrected = auto_white_balance(frame_enhanced)
    rgb_image = cv2.cvtColor(frame_corrected, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        # 選擇最中心的臉
        if len(results.multi_face_landmarks) > 1:
            img_h, img_w = img.shape[:2]
            center_x, center_y = img_w / 2, img_h / 2
            face_landmarks = min(results.multi_face_landmarks, 
                                key=lambda f: sum((p.x * img_w - center_x)**2 + (p.y * img_h - center_y)**2 
                                                for p in f.landmark))
        else:
            face_landmarks = results.multi_face_landmarks[0]

        img_h, img_w, _ = img.shape
        mesh_points = [(int(pt.x * img_w), int(pt.y * img_h), pt.z) for pt in face_landmarks.landmark]

        # Five Feature Analysis
        xs = [p[0] for p in mesh_points]
        ys = [p[1] for p in mesh_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        face_width = x_max - x_min
        face_height = y_max - y_min
        aspect_ratio = face_height / face_width if face_width != 0 else 0

        top_region_points = [p for p in mesh_points if p[1] < y_min + 0.2 * face_height]
        bottom_region_points = [p for p in mesh_points if p[1] > y_max - 0.2 * face_height]
        forehead_width = max([p[0] for p in top_region_points]) - min([p[0] for p in top_region_points]) if top_region_points else face_width
        jaw_width = max([p[0] for p in bottom_region_points]) - min([p[0] for p in bottom_region_points]) if bottom_region_points else face_width

        if aspect_ratio > 1.3:
            face_shape = "長形臉"
        elif forehead_width > jaw_width * 1.1:
            face_shape = "心形臉"
        elif jaw_width > forehead_width * 1.1:
            face_shape = "三角形臉"
        else:
            if aspect_ratio > 1.1:
                face_shape = "橢圓形臉"
            else:
                bottom_points = [p for p in mesh_points if p[1] > y_max - 0.1 * face_height]
                custom_points = bottom_points
                if len(custom_points) >= 2:
                    left_bottom = min(custom_points, key=lambda p: p[0])
                    right_bottom = max(custom_points, key=lambda p: p[0])
                    avg_bottom_y = (left_bottom[1] + right_bottom[1]) / 2.0
                    chin_y = y_max
                    if chin_y - avg_bottom_y < 0.05 * face_height:
                        face_shape = "方形臉"
                    else:
                        face_shape = "圓形臉"
                else:
                    face_shape = "圓形臉"

        # Skin Tone Analysis
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        face_oval_points = np.array([mesh_points[i][:2] for i in FACE_OVAL], dtype=np.int32)
        cv2.fillPoly(mask, [face_oval_points], 255)
        mean_color = cv2.mean(frame_corrected, mask=mask)
        avg_bgr_color = np.uint8([[mean_color[:3]]])
        avg_hsv_color = cv2.cvtColor(avg_bgr_color, cv2.COLOR_BGR2HSV)[0][0]
        hue = avg_hsv_color[0]
        
        if hue < 15 or hue > 165:
            tone = "冷膚色"
            if avg_hsv_color[2] > 130:
                season_type = "夏季型 (冷色系柔和)"
                color_suggestions = "霧藍色、薰衣草紫、粉櫻花色、薄荷綠等柔和冷色"
                update_lipstick_suggestions("冷膚色", "夏季型")
            else:
                season_type = "冬季型 (冷色系強烈)"
                color_suggestions = "寶藍色、紫羅蘭色、純白色、漆黑色等高對比冷色"
                update_lipstick_suggestions("冷膚色", "冬季型")
        elif hue < 45:
            tone = "暖膚色"
            if avg_hsv_color[2] > 130:
                season_type = "春季型 (暖色系明亮)"
                color_suggestions = "珊瑚色、鮭粉色、亮金色、淺橘色等明亮暖色"
                update_lipstick_suggestions("暖膚色", "春季型")
            else:
                season_type = "秋季型 (暖色系深沈)"
                color_suggestions = "橄欖綠、磚紅色、芥末黃、卡其色等沈穩暖色"
                update_lipstick_suggestions("暖膚色", "秋季型")
        else:
            tone = "中性膚色"
            if avg_hsv_color[2] > 130:
                season_type = "夏季型 (冷色系柔和)"
                color_suggestions = "霧藍色、薰衣草紫、粉櫻花色、薄荷綠等柔和冷色"
            else:
                season_type = "冬季型 (冷色系強烈)"
                color_suggestions = "寶藍色、紫羅蘭色、純白色、漆黑色等高對比冷色"
            update_lipstick_suggestions("中性膚色", "通用")

        # Nose Shape Analysis
        left_eye_pts = [mesh_points[i] for i in LEFT_EYE]
        right_eye_pts = [mesh_points[i] for i in RIGHT_EYE]
        left_eye_inner = min(left_eye_pts, key=lambda p: p[0])
        right_eye_inner = max(right_eye_pts, key=lambda p: p[0])
        nose_bridge_x = (left_eye_inner[0] + right_eye_inner[0]) // 2
        nose_bridge_y = (left_eye_inner[1] + right_eye_inner[1]) // 2
        nose_root = (nose_bridge_x, nose_bridge_y)
        nose_tip = mesh_points[1][:2]
        nose_length = math.hypot(nose_tip[0] - nose_root[0], nose_tip[1] - nose_root[1])
        tip_y = nose_tip[1]
        nose_band = [p for p in mesh_points if abs(p[1] - tip_y) < face_height * 0.05]
        if nose_band:
            nose_left = min(nose_band, key=lambda p: p[0])
            nose_right = max(nose_band, key=lambda p: p[0])
            nose_width = math.hypot(nose_right[0] - nose_left[0], nose_right[1] - nose_left[1])
            if nose_width > face_width * 0.15:
                nose_shape = "寬鼻"
            elif nose_length > face_height * 0.2:
                nose_shape = "長鼻"
            else:
                nose_shape = "標準鼻"

        # Eye Shape Analysis
        left_eye_height = max([p[1] for p in left_eye_pts]) - min([p[1] for p in left_eye_pts])
        right_eye_height = max([p[1] for p in right_eye_pts]) - min([p[1] for p in right_eye_pts])
        left_eye_outer = max(left_eye_pts, key=lambda p: p[0])
        right_eye_inner = min(right_eye_pts, key=lambda p: p[0])
        eye_angle_left = math.degrees(math.atan2(left_eye_outer[1] - left_eye_inner[1], left_eye_outer[0] - left_eye_inner[0]))
        eye_angle_right = math.degrees(math.atan2(right_eye_inner[1] - right_eye_pts[0][1], right_eye_inner[0] - right_eye_pts[0][0]))
        if eye_angle_left > 5 or eye_angle_right > 5:
            eye_shape = "上揚眼"
        elif eye_angle_left < -5 or eye_angle_right < -5:
            eye_shape = "下垂眼"
        else:
            eye_shape = "標準眼"

        # Brow Shape Analysis
        left_brow_pts = [mesh_points[i] for i in LEFT_EYEBROW]
        right_brow_pts = [mesh_points[i] for i in RIGHT_EYEBROW]
        left_brow_height = max([p[1] for p in left_brow_pts]) - min([p[1] for p in left_brow_pts])
        right_brow_height = max([p[1] for p in right_brow_pts]) - min([p[1] for p in right_brow_pts])
        if left_brow_height > face_height * 0.05 or right_brow_height > face_height * 0.05:
            brow_shape = "濃眉"
        else:
            brow_shape = "細眉"

        # 妝容建議
        if tone == "冷膚色":
            foundation_suggestion = "選擇偏粉調或象牙色粉底，使膚色勻稱"
            lipstick_suggestion = "適合玫紅、梅子色等冷調口紅色號"
            eyeshadow_suggestion = "適合灰棕、紫色系眼影，可選玫瑰棕等冷色號"
            blush_suggestion = "選擇粉紅或冷桃色腮紅營造氣色"
        elif tone == "暖膚色":
            foundation_suggestion = "選擇偏黃調粉底（如自然色），貼合膚色"
            lipstick_suggestion = "適合珊瑚橘、磚紅等暖色口紅色號"
            eyeshadow_suggestion = "適合大地色、古銅色等暖色眼影"
            blush_suggestion = "使用珊瑚色或蜜桃色腮紅增添氣色"
        else:
            foundation_suggestion = "選擇自然色粉底，提升膚色均勻度"
            lipstick_suggestion = "口紅可選經典紅或豆沙色，百搭色調"
            eyeshadow_suggestion = "眼影可選棕色系，自然凸顯眼神"
            blush_suggestion = "使用淡玫瑰色腮紅，自然修飾氣色"

        if nose_shape == "寬鼻":
            nose_shadow_suggestion = "在鼻梁兩側掃上陰影，使鼻子看起來更窄更立體"
        elif nose_shape == "長鼻":
            nose_shadow_suggestion = "在鼻尖下方掃陰影縮短鼻長，鼻樑不要過度打亮"
        else:
            nose_shadow_suggestion = "於鼻樑輕掃陰影與高光，自然修飾鼻型"

        if eye_shape == "上揚眼":
            eyeliner_suggestion = "順著眼型畫細眼線，眼尾稍微平拉，避免再上揚"
        elif eye_shape == "下垂眼":
            eyeliner_suggestion = "眼線在眼尾處上揚加長，打造提拉效果"
        else:
            eyeliner_suggestion = "沿著眼型畫自然眼線，眼尾略微拉長即可"

        if face_shape == "圓形臉":
            style_suggestion = "建議選擇有縱向視覺效果的領型，如 V 領或方領上衣，拉長臉部比例，避免圓領。"
        elif face_shape == "方形臉":
            style_suggestion = "可嘗試圓領、V 領或甜心領這類較柔和的領口，柔化較為方正的下顎線條，避免方領設計。"
        elif face_shape == "長形臉":
            style_suggestion = "適合寬領、圓領或高領等較寬的領口，增加臉部橫向視覺，縮短臉型比例，深V領可能讓臉看起來更長，需避免。"
        elif face_shape == "橢圓形臉":
            style_suggestion = "臉型較為均衡，多數領型皆適合。可選擇方領或圓領來突出頸部線條，此臉型可大膽嘗試各種風格。"
        elif face_shape == "心形臉":
            style_suggestion = "額頭較寬下巴尖，建議V領以轉移視覺重心，下半身搭配有墜感的服飾平衡比例；也可嘗試方領或一字領擴展下顎視覺寬度。"
        elif face_shape == "三角形臉":
            style_suggestion = "下顎較寬額頭較窄，可選擇圓領或一字領來縮小下半部視覺份量，上衣可有肩部裝飾以平衡臉部比例。"
        else:
            style_suggestion = "臉型獨特，建議嘗試各種領型找到最適合自己的風格。"

        # Line Notify
        cv2.imwrite("snapshot.jpg", frame_corrected)
        token = "wbznOI8mD3veZvQPUbmRkUP5wX5Cm5tRxBOkKaUfXOm"
        notify_url = "https://notify-api.line.me/api/notify"
        headers = {"Authorization": "Bearer " + token}
        message = (
            f"五官分析結果：\n"
            f"臉型：{face_shape}\n"
            f"眉型：{brow_shape}\n"
            f"眼型：{eye_shape}\n"
            f"鼻型：{nose_shape}\n"
            f"膚色：{tone}\n"
            f"季型：{season_type}\n"
            f"適合的服裝顏色：{color_suggestions}\n"
            f"服裝風格建議：{style_suggestion}\n"
            f"個性化妝容建議：\n"
            f"粉底：{foundation_suggestion}\n"
            f"鼻影：{nose_shadow_suggestion}\n"
            f"腮紅：{blush_suggestion}\n"
            f"眼影：{eyeshadow_suggestion}\n"
            f"眼線：{eyeliner_suggestion}\n"
            f"口紅：{lipstick_suggestion}"
        )
        payload = {"message": message}
        files = {"imageFile": open("snapshot.jpg", "rb")}
        try:
            response = requests.post(notify_url, headers=headers, data=payload, files=files)
            files["imageFile"].close()
            notify_status = "Line Notify 通知已發送" if response.status_code == 200 else f"Line Notify 發送失敗，狀態碼：{response.status_code}"
        except Exception as e:
            files["imageFile"].close()
            notify_status = f"Line Notify 發送出錯: {e}"

        # 返回結果
        return jsonify({
            "face_shape": face_shape,
            "brow_shape": brow_shape,
            "eye_shape": eye_shape,
            "nose_shape": nose_shape,
            "tone": tone,
            "season_type": season_type,
            "color_suggestions": color_suggestions,
            "style_suggestion": style_suggestion,
            "foundation_suggestion": foundation_suggestion,
            "nose_shadow_suggestion": nose_shadow_suggestion,
            "blush_suggestion": blush_suggestion,
            "eyeshadow_suggestion": eyeshadow_suggestion,
            "eyeliner_suggestion": eyeliner_suggestion,
            "lipstick_suggestion": lipstick_suggestion,
            "suggested_lipsticks": suggested_lipsticks,
            "notify_status": notify_status
        })
    return jsonify({"error": "未偵測到人臉"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)