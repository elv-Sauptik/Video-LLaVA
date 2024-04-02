import os
from typing import List, Tuple
import cv2
import tqdm


def construct_shot_dummy_mp4(object_id: str, content_part_dir: str, shots: List[Tuple[int, int]], fps: float):
    # make a dir for storing the shots contents
    dummy_video_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tmp", object_id)
    if not os.path.exists(dummy_video_path):
        os.makedirs(dummy_video_path)

    # get the parts
    video_parts = [os.path.join(content_part_dir, file)
                   for file in sorted(os.listdir(content_part_dir))]

    # convert the shot start and end from millisecond to frame index
    shots = [[s * fps // 1000, e * fps / 1000] for s, e in shots]

    # shot cnt
    cnt = -1

    # current frame index
    idx = 0

    # shot buffer
    cur_s = shots[0][0]
    cur_e = shots[0][1]
    shot = []

    # loop all files
    for file in tqdm.tqdm(video_parts):
        # frame buffer and indices buffer
        frames = []
        indices = []
        # read video
        cap = cv2.VideoCapture(file)
        # check fps
        _fps = cap.get(5)
        assert fps == _fps
        # catch all frames
        while (True):
            ret, frame = cap.read()
            if (ret):
                frames.append(frame)
                indices.append(idx)
                idx += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # construct shots using the frames
        for i, f in zip(indices, frames):
            if cur_s < i <= cur_e:
                shot.append(f)
            elif i > cur_e:
                # write the shot into mp4 file
                cnt += 1
                shot_file = os.path.join(dummy_video_path, f"shot_{cnt}.mp4")
                # dump shot to mp4
                output = cv2.VideoWriter(
                    shot_file,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (shot[0].shape[1], shot[0].shape[0])
                )
                for shot_frame in shot:
                    output.write(shot_frame)
                output.release()

                # update shot info
                shots.pop(0)
                cur_s = shots[0][0]
                cur_e = shots[0][1]
                shot = [f]
            else:
                continue

        # release cap
        cap.release()

    # clean
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import json
    object_id = "iq__3BCpFm5NquAuzCMqekkZX8rR3hF"
    content_part_dir = "/ml/sony_movies/iq__3BCpFm5NquAuzCMqekkZX8rR3hF"
    fps = 24000/1001
    input(f"Check fps for {object_id} is {fps}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "shot_info", f"{object_id}.json")) as f:
        shot_track = json.load(f)

    shots = []
    for part_idx in shot_track[object_id]:
        for shot in shot_track[object_id][part_idx]:
            shots.append([shot["start_time"], shot["end_time"]])

    print(len(shots), "shots in total")

    construct_shot_dummy_mp4(object_id, content_part_dir, shots, fps)
