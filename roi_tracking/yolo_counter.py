"""
Скрипт для подсчёта автомобилей в видеопотоке с помощью YOLO и OpenCV.

Функциональность:
- Проигрывает видео и позволяет пользователю указать прямоугольную область интереса (ROI),
  нарисовав её мышкой на выбранном кадре.
- После выбора ROI запускает трекинг объектов (машин) с использованием YOLO.
- Считает автомобили, центр которых попадает в заданную область.
- Сохраняет аннотированное видео с боксами, ROI и счётчиком машин.

Управление:
- Во время предварительного просмотра: ПРОБЕЛ — остановка и переход к выбору ROI.
- В режиме выбора ROI: нарисовать прямоугольник мышью, затем ENTER / ПРОБЕЛ — подтвердить,
  ESC — отменить.
- Во время трекинга и подсчёта: ESC — завершить работу.
"""

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# ========== Настройки ==========

# Путь к входному видеофайлу
video_path = "input.mp4"

# Путь к весам модели YOLO
model_path = "yolo11x.pt"

# Путь к выходному видео
output_path = "out_counted.mp4"

# На каком кадре по умолчанию остановиться, если пользователь сам не нажал паузу
DEFAULT_SELECT_FRAME = 100


# ========== Функция: точка в полигоне ==========

def is_point_in_polygon(x, y, polygon_points):
    """
    Проверяет, находится ли точка (x, y) внутри полигона или на его границе.

    Использует:
        - cv2.convexHull для получения выпуклой оболочки полигона
        - cv2.pointPolygonTest для определения положения точки

    :param x: координата X точки
    :param y: координата Y точки
    :param polygon_points: numpy-массив точек полигона формы (N, 2)
    :return: True, если точка внутри полигона или на его границе, иначе False
    """
    contour = cv2.convexHull(polygon_points)
    dist = cv2.pointPolygonTest(contour, (x, y), False)
    return dist >= 0  # >= 0 => внутри или на границе


# ========== Фаза 1: проигрываем видео и выбираем ROI на кадре ==========

def play_and_select_roi(video_path, select_frame=DEFAULT_SELECT_FRAME):
    """
    Проигрывает видео до указанного кадра (или до нажатия пробела) и позволяет
    пользователю выделить прямоугольную область интереса (ROI) мышкой.

    Этапы:
        1. Режим "play": показ кадров, ожидание достижения select_frame либо нажатия пробела/'p'.
        2. Режим "select": на замороженном кадре пользователь рисует прямоугольник ЛКМ.
           - После отпускания кнопки мыши прямоугольник фиксируется.
           - ENTER / ПРОБЕЛ / 's' — подтверждение выбора.
           - ESC — отмена выбора.

    :param video_path: путь к видеофайлу
    :param select_frame: номер кадра, на котором автоматически остановиться,
                         если пользователь не нажал паузу раньше
    :return: (region_polygon, frame_idx)
             region_polygon — numpy-массив из 4 точек (x, y), описывающих прямоугольник,
             frame_idx — номер кадра, на котором был выбран ROI.
             В случае ошибки или отмены — (None, None).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка при открытии {video_path}")
        return None, None

    window_name = "Counting Cars"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    phase = "play"   # "play" -> крутим видео; "select" -> рисуем ROI на замороженном кадре

    pause_frame = None
    frame = None

    # Данные для рисования прямоугольника
    roi_data = {
        "drawing": False,
        "ix": -1, "iy": -1,
        "ex": -1, "ey": -1,
        "done": False
    }

    def mouse_callback(event, x, y, flags, param):
        """
        Обработчик событий мыши для режима выбора ROI.
        Работает только, когда phase == "select".
        """
        nonlocal frame, pause_frame
        if phase != "select":
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            roi_data["drawing"] = True
            roi_data["ix"], roi_data["iy"] = x, y

        elif event == cv2.EVENT_MOUSEMOVE and roi_data["drawing"]:
            # Рисуем прямоугольник поверх замороженного кадра
            frame[:] = pause_frame.copy()
            cv2.rectangle(frame, (roi_data["ix"], roi_data["iy"]), (x, y), (0, 255, 255), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            roi_data["drawing"] = False
            roi_data["ex"], roi_data["ey"] = x, y
            frame[:] = pause_frame.copy()
            cv2.rectangle(
                frame,
                (roi_data["ix"], roi_data["iy"]),
                (roi_data["ex"], roi_data["ey"]),
                (0, 255, 255),
                2
            )
            roi_data["done"] = True

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        if phase == "play":
            # Режим проигрывания видео
            ret, frame = cap.read()
            if not ret:
                print("Видео закончилось раньше, чем выбрали ROI.")
                break

            frame_idx += 1

            # Подсказка пользователю
            cv2.putText(
                frame,
                "Пробел - определить область",
                (30, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF  # скорость воспроизведения

            # Условия паузы:
            #  - дошли до заранее заданного кадра
            #  - или пользователь нажал пробел / 'p'
            if frame_idx >= select_frame or key in (32, ord('p')):
                phase = "select"
                pause_frame = frame.copy()
                frame = pause_frame.copy()  # будем рисовать прямо на этом кадре

        elif phase == "select":
            # Режим выбора области интереса
            cv2.putText(
                frame,
                "Нарисуйте область исследования",
                (30, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 255, 255),
                2
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # Подтверждение выбора
            if roi_data["done"] and key in (13, 32, ord('s')):  # Enter, Space, 's'
                # Нормализуем координаты прямоугольника (x_min < x_max, y_min < y_max)
                x1, y1 = roi_data["ix"], roi_data["iy"]
                x2, y2 = roi_data["ex"], roi_data["ey"]
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])

                region_polygon = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.int32)

                cv2.destroyWindow(window_name)
                cap.release()
                print(f"ROI выбран на кадре {frame_idx}")
                return region_polygon, frame_idx

            # Отмена
            if key == 27:  # Esc
                cv2.destroyWindow(window_name)
                cap.release()
                print("Выбор ROI отменён.")
                return None, None

    cv2.destroyWindow(window_name)
    cap.release()
    return None, None


# ========== Основной код ==========

# 1) Сначала выбираем ROI на нужном кадре
region_polygon, start_frame_idx = play_and_select_roi(video_path, select_frame=DEFAULT_SELECT_FRAME)
if region_polygon is None:
    print("ROI не выбран, выхожу.")
    exit()

# 2) Инициализируем модель YOLO
model = YOLO(model_path)

# 3) Открываем видео для трекинга и подсчёта, начиная с кадра, где мы выбрали ROI
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Ошибка при открытии {video_path}")
    exit()

# Переходим к кадру, на котором остановились (или рядом с ним)
# -1 чтобы на всякий случай захватить тот же кадр
if start_frame_idx is not None and start_frame_idx > 1:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx - 1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Хранение истории треков (пока не используется), и флагов "засчитан / не засчитан"
track_history = defaultdict(lambda: [])
has_been_counted = defaultdict(lambda: False)  # Для каждого track_id: был ли засчитан
car_count = 0

window_name = "Counting Cars"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 4) Основной цикл трекинга и подсчёта
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Видео закончено или произошла ошибка чтения.")
        break

    # Запуск трекинга
    results = model.track(frame, persist=True)
    annotated_frame = frame.copy()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # Рисуем боксы/лейблы от Ultralytics (тонкие рамки и мелкий шрифт)
        annotated_frame = results[0].plot(line_width=1, font_size=0.4)

        for box, track_id, cls_idx, conf in zip(boxes, track_ids, cls_indices, confidences):
            x_c, y_c, w, h = box
            class_name = results[0].names[cls_idx]

            # Считаем только машины (можно расширить до truck/bus и т.д.)
            if class_name == 'car':
                center_x = float(x_c)
                center_y = float(y_c)

                inside_zone = is_point_in_polygon(center_x, center_y, region_polygon)

                # Если объект впервые попал в зону => считаем
                if inside_zone and not has_been_counted[track_id]:
                    car_count += 1
                    has_been_counted[track_id] = True

    # Текст с количеством машин
    cv2.putText(
        annotated_frame,
        f"Машины с Патриков: {car_count}",
        (50, 740),
        cv2.FONT_HERSHEY_COMPLEX,
        1.2,
        (0, 0, 255),
        2
    )

    # Рисуем ROI
    cv2.polylines(
        annotated_frame,
        [region_polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=1
    )

    cv2.imshow(window_name, annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) == 27:  # Esc
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Общее количество автомобилей, проехавших по выделенной полосе: {car_count}")
