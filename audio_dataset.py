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
from pydub import AudioSegment
import moviepy.editor as mp
warnings.filterwarnings("ignore")

# Run using CUDA_VISIBLE_DEVICES=0 python3 audio_dataset.py --config_file biden.csv --device cuda --device_ids 0 --download

def download_audio_as_wav(url, video_id, person, dl_path):
    """
    Downloads the audio of a YouTube video and saves it as a .wav file
    """
    filename = f"{person}-{video_id}"

    yt = YouTube(url)
    try:
        audio = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        audio_file_path = audio.download(filename=filename+".mp4", output_path=dl_path)
        audio = AudioSegment.from_file(audio_file_path, format="mp4")
        audio.export(filename+".wav", format="wav")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



def extract_audio_segment(video_id, person, ts, te, dl_path='.\dataset'):
    """
    Extracts a segment of an audio .wav file
    """
    filename = f"{person}-{video_id}"
    audio_file_path = os.path.join(dl_path, filename+".wav")

    # Check if the audio file exists
    assert os.path.exists(audio_file_path), f"Audio file, {audio_file_path}, does not exist, probably not downloaded"
    
    # Load the audio
    audio = AudioSegment.from_wav(audio_file_path)
    
    # Convert timestamps to milliseconds
    ts_ms = int(ts * 1000)
    te_ms = int(te * 1000)

    # Extract audio segment
    audio_sub = audio[ts_ms:te_ms]

    return audio_sub, filename


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

    parser.add_argument('--path', default="dataset_biden/")
    parser.add_argument('--config_file', default="biden.csv")
    parser.add_argument('--extension', default=".wav")
    parser.add_argument('--download', dest="dl", action="store_true", help="Whether to download")
    parser.add_argument('--multi', dest="multi", action="store_true", help="Give 'multi' to download using multiprocessing")
    parser.set_defaults(dl=False)
    parser.set_defaults(multi=False)

    args = parser.parse_args()

    config = args.config_file
    youtube = "https://www.youtube.com/watch?v="
    dl_path = args.path
    ext = args.extension
    out_size = (256, 256)
    
    if not os.path.exists(dl_path):
        os.mkdir(dl_path)


    # Read the config
    df = pd.read_csv(config)
    # Trim the start and end second
    df = trim_row(df=df)
    # Set number for repetative videos
    df["num"] = df.groupby('video_id').cumcount()

    # Download the videos
    df['video_url'] = youtube + df['video_id']
    
    if args.dl:
        try:
            if args.multi:
                # Simultaniously
                print("STart Downloading...")
                pool = multiprocessing.Pool(processes=len(df))
                pool.starmap(download_audio_as_wav,
                            [(df.iloc[i]['video_url'], df.iloc[i]['video_id'], df.iloc[i]['person_id'], dl_path) for i in range(len(df))])
                pool.close()
                pool.join()
            else:
                # One by One
                print("STart Downloading...")
                for i in tqdm.trange(len(df)):
                    download_audio_as_wav(df.iloc[i]['video_url'], df.iloc[i]['video_id'], df.iloc[i]['person_id'], dl_path)
        except Exception as e:
            print(e)

    df.drop(['video_url'], axis=1, inplace=True)


    #TODO: Check how long each subseg should be
    #TODO: Final Check
    # Preprocessing
    print("Preprocessing starts...")
    pbar = tqdm.tqdm(range(len(df)))
    for i in tqdm.tqdm(range(len(df))):
        video_id, ts, te, partition, person, num = df.iloc[i]
        data_path = os.path.join(dl_path, partition, person, video_id)
        os.makedirs(data_path, exist_ok=True)

        # Cut the video segment defined in the config
        audio_sub, filename = extract_audio_segment(video_id=video_id, person=person, 
                                                       ts=ts, te=te, dl_path=dl_path)
             
        audio_sub.export(os.path.join(data_path, f"{filename}_{num}.wav"), format="wav")

    print("Successfully Done!")