import os
import cv2
import sys, getopt
import argparse




# Добавление парсинга аргументов командной строки
parser = argparse.ArgumentParser(description="Process a video file.")

parser.add_argument("--video_path", type=str, help="Path to the video file")
parser.add_argument('--csv_path', type=str, default='',
                    help='load csv have labeled')
parser.add_argument('--fullscreen', action='store_true', default=False,
                    help='Open labeling window in fullscreen mode')
args = parser.parse_args()

video_path = args.video_path  # Получение пути к видеофайлу из аргументов командной строки

if not video_path or not os.path.isfile(video_path) or not video_path.endswith('.mp4'):
    print("Not a valid video path! Please modify path in parser.py --label_video_path")
    sys.exit(1)

# create information record dictionary
# Frame: index of frame
# Ball : 0 for no ball or not clearly visible, 1 for having ball
# x: x position of ball center
# y: y position of ball center
csv_path = args.csv_path



def save_info(info, video_path):
    success = False
    video_name = 'csv/' + os.path.split(video_path)[-1][:-4]
    csv_path = video_name+'_ball.csv'
    try:
        with open(csv_path, 'w') as file:
            file.write("Frame,Visibility,X,Y\n")
            for frame in info:
                data = "{},{},{:.3f},{:.3f}".format(info[frame]["Frame"], info[frame]["Visibility"],
                                            info[frame]["X"],info[frame]["Y"])
                file.write(data+'\n')
        success = True
        print("Save info successfully into", video_name+'_ball.csv')
    except:
        print("Save info failure ", csv_path)

    return success

def load_info(csv_path):
    with open(csv_path, 'r') as file:
        lines = file.readlines()
        n_frames = len(lines) - 1
        info = {
            idx:{
            'Frame': idx,
            'Visibility': 0,
            'x': -1,
            'y': -1
            } for idx in range(n_frames)
        }

        for line in lines[1:]:
            frame, Visibility, x, y = line.split(',')[0:4]
            frame = int(frame)

            if info.get(frame) is None:
                print("Frame {} not found in info, creating new entry.".format(frame))
                info[frame] = {
                    'Frame': frame,
                    'Visibility': 0,
                    'X': -1,
                    'Y': -1
                }


            info[frame]['Frame'] = frame
            info[frame]['Visibility'] = int(Visibility)
            info[frame]['X'] = float(x)
            info[frame]['Y'] = float(y)

    return info

def show_image(image, frame_no, x, y):
    h, w, _ = image.shape
    if x != -1 and y != -1:
        x_pos = int(x)
        y_pos = int(y)
        cv2.circle(image, (x_pos, y_pos), 5, (0, 0, 255), -1)
    
    # Calculate dynamic font size based on video resolution
    # Base resolution: 1920x1080 -> font_scale=2.0, thickness=3
    # Scale proportionally for other resolutions
    base_width, base_height = 1920, 1080
    base_font_scale = 2.0
    base_thickness = 3
    
    # Calculate scaling factor based on image area ratio
    current_area = w * h
    base_area = base_width * base_height
    scale_factor = (current_area / base_area) ** 0.5  # Square root for linear dimension scaling
    
    # Apply minimum and maximum limits
    scale_factor = max(0.3, min(1.0, scale_factor))  # Limit between 0.3x and 1.0x
    
    font_scale = base_font_scale * scale_factor
    thickness = max(1, int(base_thickness * scale_factor))
    
    # Adjust text position based on scaled font size
    text_x = max(15, int(30 * scale_factor))
    text_y = max(30, int(60 * scale_factor))
    
    # Font scaling is working correctly - values calculated above
    
    text = "Frame: {}".format(frame_no)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    return image

def go2frame(cap, frame_no, info):
    x,y = -1, -1
    if frame_no in info:
        x, y = info[frame_no]['X'], info[frame_no]['Y']

    cap.set(1, frame_no)
    ret, image = cap.read()
    image = show_image(image, frame_no, x, y)
    return image


load_csv = False
if os.path.isfile(csv_path) and csv_path.endswith('.csv'):
    load_csv = True
else:
    print("Not a valid csv file! Please modify path in parser.py --csv_path")

# acquire video info
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# import pdb

# pdb.set_trace()

if load_csv:
    info = load_info(csv_path)
    if  len(info) != n_frames:
        print("Number of frames in video and dictionary are not the same!")
        print("Fail to load, create new dictionary instead.")
        info = {
            idx: {"Frame": idx, "Visibility": 0, "X": -1, "Y": -1}
            for idx in range(n_frames)
        }
    else:
        print("Load labeled dictionary successfully.")
else:
    print("Create new dictionary")
    info = {
        idx: {"Frame": idx, "Visibility": 0, "X": -1, "Y": -1}
        for idx in range(n_frames)
    }

# # # # # # # # # # # # # # # #
# e: exit program             #
# s: save info                #
# n: next frame               #
# p: previous frame           #
# f: to first frame           #
# l: to last frame            #
# >: fast forward 36 frames   #
# <: fast backward 36 frames  #
# F/f: toggle fullscreen      #
# # # # # # # # # # # # # # # #

def ball_label(event, x, y, flags, param):
    global frame_no, info, image

    if not frame_no in info:
        print("Frame {} not found in info, creating new entry.".format(frame_no))
        info[frame_no] = {
            'Frame': frame_no,
            'Visibility': 0,
            'X': -1,
            'Y': -1
        }

    if event == cv2.EVENT_LBUTTONDOWN:
        h, w, _ = image.shape
        info[frame_no]['X'] = x
        info[frame_no]['Y'] = y
        info[frame_no]["Visibility"] = 1

    elif event == cv2.EVENT_MBUTTONDOWN:
        info[frame_no]['X'] = -1
        info[frame_no]['Y'] = -1
        info[frame_no]["Visibility"] = 0

saved_success = False
frame_no = 0
_, image = cap.read()


show_image(image, 0, info[0]['X'], info[0]['Y'])

# Create window with appropriate flags
window_name = 'Ball Labeling'
fullscreen_mode = args.fullscreen  # Track current fullscreen state

if fullscreen_mode:
    cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
    print("Window opened in fullscreen mode. Press 'F' to toggle fullscreen/offscreen.")
else:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Window opened in normal mode. Press 'F' to toggle fullscreen.")
while True:
    leave = 'y'
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, ball_label)
    key = cv2.waitKey(1) & 0xFF
    
    # Handle fullscreen toggle
    if key == ord('F') or key == ord('f'):
        # Auto-save before toggling fullscreen to prevent data loss
        temp_saved = save_info(info, video_path)
        if temp_saved:
            print("Auto-saved current annotations before fullscreen toggle")
        
        # Save current frame position for restoration
        saved_frame_no = frame_no
        
        if fullscreen_mode:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            fullscreen_mode = False
            print("Exited fullscreen mode")
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            fullscreen_mode = True
            print("Entered fullscreen mode")
        
        # Restore frame position and refresh display
        frame_no = saved_frame_no
        image = go2frame(cap, frame_no, info)
        print(f"Restored frame {frame_no} after fullscreen toggle")
    
    if key == ord('e'):
        if not saved_success:
            print("You forget to save file!")
            while True:
                leave = str(input("Really want to leave without saving? [Y/N]"))
                leave = leave.lower()
                if leave != 'y' and leave != 'n':
                    print("Please type 'y/Y' or 'n/N'")
                    continue
                elif leave == 'y':
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Exit label program")
                    sys.exit(1)
                elif leave == 'n':
                    break

        if leave == 'y':
            cap.release()
            cv2.destroyAllWindows()
            print("Exit label program")
            sys.exit(1)

    elif key == ord('s'):
        saved_success = save_info(info, video_path)

    elif key == ord('n'):
        if frame_no >= n_frames-1:
            print("This is the last frame")
            continue
        frame_no += 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('p'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no -= 1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('f'):
        if frame_no == 0:
            print("This is the first frame")
            continue
        frame_no = 0
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('l'):
        if frame_no == n_frames-1:
            print("This is the last frame")
            continue
        frame_no = n_frames-1
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('>'):
        if frame_no + 36 >= n_frames-1:
            print("Reach last frame")
            frame_no = n_frames-1
        else:
            frame_no += 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))

    elif key == ord('<'):
        if frame_no - 36 <= 0:
            print("Reach first frame")
            frame_no = 0
        else:
            frame_no -= 36
        image = go2frame(cap, frame_no, info)
        print("Frame No.{}".format(frame_no))
    else:
        image = go2frame(cap, frame_no, info)
