import numpy as np

THRESH_FACE = 3.0   # mais permissivo
THRESH_HAND = 1.5   # mais restritivo

def detect_objects(frame, model, nose_point, eye_l, eye_r, hands):
    phone_center = None
    hands_busy = False

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in ["cell phone", "bottle", "cup"]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_center = np.array([(x1 + x2)//2, (y1 + y2)//2])

            # ============================
            # salvar celular
            # ============================
            if label == "cell phone":
                phone_center = tuple(obj_center)

            # ============================
            # validação geométrica
            # ============================
            if nose_point and eye_l and eye_r and hands:

                nose = np.array(nose_point)
                eye_l_np = np.array(eye_l)
                eye_r_np = np.array(eye_r)

                # normalização
                d_eye = np.linalg.norm(eye_l_np - eye_r_np)
                if d_eye == 0:
                    continue

                # ----------------------------
                # distância ao rosto
                # ----------------------------
                d_face = np.linalg.norm(obj_center - nose) / d_eye

                close_to_face = d_face < THRESH_FACE

                # ----------------------------
                # distância às mãos
                # ----------------------------
                close_to_hand = False

                for hand in hands:
                    hand_np = np.array(hand)
                    d_hand = np.linalg.norm(obj_center - hand_np) / d_eye

                    if d_hand < THRESH_HAND:
                        close_to_hand = True
                        break

                # ----------------------------
                # condição final
                # ----------------------------
                if close_to_face and close_to_hand:
                    hands_busy = True

    return phone_center, hands_busy
