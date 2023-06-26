import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from argparse import ArgumentParser
import multiprocessing

import cv2
import torch
from skimage.transform import resize
from face_alignment import LandmarksType, FaceAlignment
import pytube
from pytube import YouTube
import moviepy.editor as mp
warnings.filterwarnings("ignore")

# Run using python3 crop_finetune.py --config_file sadeghi-modi.csv --device cpu

def download_video(url, video_id, person, dl_path, ext='.mp4'):
    """
    Downloads the videos and cut the segments
    """
    file_extension='mp4' if ext==".mp4" else None
    filename = f"{person}-{video_id}"
    
    yt = YouTube(url)
    try:
        #video = yt.streams.filter(progressive=True, file_extension=file_extension).order_by('resolution').desc().first()
        video = yt.streams.filter(progressive=True, file_extension=file_extension).get_highest_resolution()
        video.download(filename=filename+ext, output_path=dl_path)
    except pytube.exceptions.RegexMatchError:
        print("RegexMatchError: Could not find match for multiple. Skipping this video.")



def extract_segment(video_id, person, ts, te, partition, dl_path='.\dataset', ext='.mp4'):
    """
    """
    filename = f"{person}-{video_id}"
    # Crop the video
    video = mp.VideoFileClip(os.path.join(dl_path, filename+ext))
    fps = video.fps

    video_sub = video.subclip(ts, te)

    return video_sub, filename, fps


def crop_centered_head(frame, bbox):
    
    h, w = frame.shape[:2]
    x, y, x2, y2, _ = bbox

    try:
        h, w = frame.shape[:2]

        # Calculate the width and height of the detection
        width = x2 - x
        height = y2 - y

        # Calculate the center of the detection
        center_x = x + (width // 2)
        center_y = y + (height // 2)

        # Calculate the size of the head
        head_size = int(max(width, height) * 1.25)

        # Calculate the coordinates for the head crop
        x1 = max(0, center_x - (head_size // 2))
        y1 = max(0, center_y - (head_size // 2))
        x2 = min(w, x1 + head_size)
        y2 = min(h, y1 + head_size)

        # Crop the head
        head = frame[y1:y2, x1:x2]
    except IndexError:
        print("Check Code! centering and cropping does not work")

    return head

def trim_row(df):
    df['diff'] = df['end'] - df['start']

    new_df = []
    for _, row in df.iterrows():
        if row['diff'] > 5:
            num_crops = row['diff'] // 5
            res = row['diff'] % 5

            for i in range(num_crops):
                clip_start = row['start'] + i * 5
                clip_end = min(row['start'] + (i + 1) * 5, row['end'])
                clip_row = row.copy()
                clip_row['start'] = clip_start
                clip_row['end'] = clip_end
                new_df.append(clip_row)

            clip_row = row.copy()
            clip_row['start'] = clip_end
            clip_row['end'] = clip_end + res
            new_df.append(clip_row)

        else:
            new_df.append(row)

    new_df = pd.DataFrame(new_df).drop('diff', axis=1)
    new_df.reset_index(drop=True, inplace=True)
    
    return new_df


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--path', default="dataset")
    parser.add_argument('--config_file', default="koushanfar.csv")
    parser.add_argument('--device', default='cpu')
    parser.add_argument("--device_ids", default="1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument('--extension', default=".mp4")
    parser.add_argument('--multi', dest="multi", action="store_true", help="Give 'multi' to download using multiprocessing")
    parser.set_defaults(multi=False)

    args = parser.parse_args()

    config = args.config_file
    youtube = "https://www.youtube.com/watch?v="
    dl_path = args.path
    ext = args.extension
    device_ids = args.device_ids
    out_size = (256, 256)
    
    if not os.path.exists(dl_path):
        os.mkdir(dl_path)
    
    if args.device == 'cuda':
        device = (f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
        if device != f'cuda:{device_ids[0]}':
            print("Couldn't find a gpu")
    else:
        device = 'cpu'

    # Read the config
    df = pd.read_csv(config)
    df["num"] = df.groupby('video_id').cumcount()

    # Download the videos
    df['video_url'] = youtube + df['video_id']
    
    if args.multi:
        # Simultaniously
        print("STart Downloading...")
        pool = multiprocessing.Pool(processes=len(df))
        pool.starmap(download_video,
                    [(df.iloc[i]['video_url'], df.iloc[i]['video_id'], df.iloc[i]['person_id'], dl_path) for i in range(len(df))])
        pool.close()
        pool.join()
    else:
        # One by One
        print("STart Downloading...")
        for i in tqdm.trange(len(df)):
            download_video(df.iloc[i]['video_url'], df.iloc[i]['video_id'], df.iloc[i]['person_id'], dl_path)

    df.drop(['video_url'], axis=1, inplace=True)

    # Trim the start and end second
    df = trim_row(df=df)

    # Initialize Face Detection
    fd = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=device).face_detector

    # Preprocessing
    print("Preprocessing starts...")
    pbar = tqdm.tqdm(range(len(df)))
    for i in tqdm.tqdm(range(len(df))):
        video_id, ts, te, partition, person, num = df.iloc[i]
        data_path = os.path.join(dl_path, partition, person, video_id+'-'+str(num))
        os.makedirs(data_path, exist_ok=True)

        # Cut the video segment defined in the config
        video_sub, filename, _ = extract_segment(video_id=video_id, person=person, ts=ts, te=te, partition=partition,
                        dl_path=dl_path)        
        save_png = True
        if not save_png:
            #### WARNING: May require large free memory ####
            #video_sub = video_sub.resize(256, 256)
            #video_sub.write_videofile(video_path, fps=fps)
            pass
        else:
            for i, frame in enumerate(video_sub.iter_frames()):
                try:
                    # Detect Face and cut the frames and save the video
                    frame_path = os.path.join(data_path, f'{filename}-{i:04d}.png')

                    bbox = fd.detect_from_image(frame)[0].astype(np.uint32)
                    head = crop_centered_head(frame, bbox)

                    # Resize to 265x256 (out_size)
                    head = resize(head, out_size, anti_aliasing=True)
                    head = head*(255/head.max())
                    head = head.clip(0, 255).astype(np.uint8)
                    mp.ImageClip(head).save_frame(frame_path)
                
                except IndexError:
                    pbar.write("IndexErr, Probably no face detected")

    print("Successfully Done!")