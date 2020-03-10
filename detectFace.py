#test
import cv2 


font = cv2.FONT_ITALIC


def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src) # 행렬 변경 
        dst = cv2.flip(dst, 1)   # 뒤집기

    elif degrees == 180:
        dst = cv2.flip(src, 0)   # 뒤집기

    elif degrees == 270:
        dst = cv2.transpose(src) # 행렬 변경
        dst = cv2.flip(dst, 0)   # 뒤집기
    else:
        dst = null
    return dst


def faceDetect():
    eye_detect = False
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")  # 얼굴찾기 haar 파일
    eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml") # 눈찾기 haar 파일

    try:
        cam = cv2.VideoCapture(0)
    except:
        print("camera loading error")
        return
    count = 0
    while True:
        ret, frame = cam.read()
        frame = Rotate(frame, 90)
        if not ret:
            break

        if eye_detect:
            info = "Eye Detention ON"
        else:
            info = "Eye Detection OFF"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #카메라 영상 왼쪽위에 위에 셋팅된 info 의 내용 출력
        cv2.putText(frame, info, (5,15), font, 0.5, (255,0, 255),1)

        for(x,y, w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)  #사각형 범위
            cv2.putText(frame, "Detected Face", (x-5, y-5), font, 0.5, (255,255,0),2)  #얼굴찾았다는 메시지
            if eye_detect:  #눈찾기
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        cv2.imshow("frame", frame)
        k=cv2.waitKey(30)

        #실행 중 키보드 i 를 누르면 눈찾기를 on, off한다.
        if k == ord('i'):
            eye_detect = not eye_detect
        if k == ord('c'):
            cv2.imwrite("frame%d.jpg" % count, frame)
            count += 1
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


faceDetect()