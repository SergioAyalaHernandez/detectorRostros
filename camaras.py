import cv2


def show_cameras(indices):
    captures = []

    for index in indices:
        cap = cv2.VideoCapture(index)
        if cap is None or not cap.isOpened():
            print(f"Error: no se puede acceder a la cámara con índice {index}")
            cap.release()
        else:
            captures.append((index, cap))

    try:
        while True:
            for idx, cap in captures:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f"Camara {idx}", frame)

            # Cierra las ventanas si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for _, cap in captures:
            cap.release()

        cv2.destroyAllWindows()


def main():
    show_cameras([0, 1, 2])


if __name__ == "__main__":
    main()
