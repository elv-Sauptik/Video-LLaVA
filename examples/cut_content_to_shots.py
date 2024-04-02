import os
from typing import List, Tuple
import cv2
import tqdm


def construct_shot_dummy_mp4(object_id: str, content_part_dir: str, shots: List[Tuple[int, int]], fps: float):
    # make a dir for storing the shots contents
    dummy_video_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tmp", object_id)
    if not os.path.exists(dummy_video_path):
        input(f"Press any key to agree to makeing new dir for {object_id}")
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

    # flag: not shot info left
    finish = False

    # loop all files
    bar = tqdm.tqdm(video_parts)
    for file in bar:
        # frame buffer and indices buffer
        frames = []
        indices = []
        # read video
        cap = cv2.VideoCapture(file)
        # check fps
        _fps = cap.get(5)
        bar.set_postfix_str(f"part {_fps}, expected fps: {fps}")
        # assert fps == _fps
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
                shot_file = os.path.join(
                    dummy_video_path, f"shot_{cnt:04}.mp4")
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
                # check if there are still some shots
                if len(shots) > 0:
                    cur_s = shots[0][0]
                    cur_e = shots[0][1]
                    shot = [f]
                else:
                    finish = True
                    break
            else:
                continue

        # release cap
        cap.release()

        # no shot left
        if finish:
            break

    # clean
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import json

    # object_id = "iq__3BCpFm5NquAuzCMqekkZX8rR3hF"
    # content_part_dir = f"/ml/sony_movies/{object_id}"
    # tags_dir = f"/ml/sony_movie_tags/{object_id}"
    # fps = 24000/1001

    # object_id = "iq__3VSr2CG4RpiqH1EsQSLEhuGLfyAx"
    # content_part_dir = f"/ml/mgm_movies/{object_id}"
    # tags_dir = f"/ml/mgm_movie_tags/{object_id}"
    # fps = 24000/1001

    # object_id = "iq__Fza2o2Xry75AMMwkSmaeyh3HJg3"
    # content_part_dir = f"/ml/uefa_matches/{object_id}"
    # tags_dir = f"/ml/uefa_match_tags/{object_id}"
    # fps = 50.0

    object_id = "iq__3WxSAwteTyum3YZ2sJYYuSkbGFq7"
    content_part_dir = f"/ml/epcr_matches/{object_id}"
    tags_dir = f"/ml/epcr_match_tags/{object_id}"
    fps = 50.0

    input(f"Check fps for {object_id} is {fps}")

    with open(os.path.join(tags_dir, "shots.json")) as f:
        shot_track = json.load(f)

    shots = []
    for shot in shot_track:
        if shot:
            shots.append([shot["start_time"], shot["end_time"]])

    print(len(shots), "shots in total")

    construct_shot_dummy_mp4(object_id, content_part_dir, shots, fps)
