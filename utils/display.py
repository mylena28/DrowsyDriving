import cv2

def get_color(score):

    if score > 80:
        return (0,255,0)

    elif score > 60:
        return (0,255,255)

    elif score > 40:
        return (0,165,255)

    elif score > 20:
        return (0,0,255)

    else:
        return (0,0,150)


def draw_status(frame, state, score, alerts=None):

    x, y = 20, 40
    color = get_color(score)

    cv2.putText(frame, state,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)

    if alerts:
        for i, alert in enumerate(alerts):
            cv2.putText(frame, f"  {alert}",
                        (x, y + 40 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 255), 2)

    return frame
