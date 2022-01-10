import data_extraction as de
import cv2
import time
import numpy as np

from mtcnn import MTCNN
from config import parser

if __name__ == "__main__":

    config = vars(parser.parse_args())

    print("Video annotation with MTCNN\n")
    print("Settings:")
    for k, v in config.items():
        print(f"{k}: {v}")

    mtcnn = MTCNN(config=config)

    cap = cv2.VideoCapture('./videos/'+ config['file'])
    out = None#cv2.VideoWriter('./videos/annotated/'+ config['file'], fourcc, 30.0, (640,480))#.split('.')[0] + ".avi"

    counter = 0
    alpha = 0.9
    avg_fps = 0

    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Check for faces
        start = time.process_time()

        scores, faces, landmarks = mtcnn(frame)

        time_elapsed = time.process_time() - start

        for i, (x,y,xf,yf) in enumerate(faces):
            frame = cv2.putText(frame, f"{int(scores[i]*100)}%", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            cv2.rectangle(frame,(x,y),(xf,yf),(255,0,0),2)

        if landmarks is not None:
            for l in landmarks:
                for i in range(0,10,2):
                    frame = cv2.circle(frame, (int(l[i]), int(l[i+1])), radius=5, color=(0, 0, 255), thickness=-1)

        current_fps = 1.0/time_elapsed
        avg_fps = alpha * avg_fps + (1.0 - alpha) * current_fps

        frame = cv2.putText(frame, f"{avg_fps:.2f}FPS", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)

        # Display the resulting frame
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS)
            res = (frame.shape[1],frame.shape[0])
            path = 'videos/annotated/'+ config['file'].split('.')[0] + '.avi'
            out = cv2.VideoWriter(path, fourcc, fps, res)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
