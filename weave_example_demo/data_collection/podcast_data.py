import time
from pathlib import Path

import pandas as pd
from langchain.document_loaders import YoutubeLoader
from pytube import Playlist, YouTube
from tqdm import tqdm

playlist_url: str = (
    "https://www.youtube.com/playlist?list=PLD80i8An1OEEb1jP0sjEyiLG8ULRXFob_"
)
root_data_dir: Path = Path("data")


def retry_access_yt_object(url, max_retries=5, interval_secs=5):
    """
    Retries creating a YouTube object with the given URL and accessing its title several times
    with a given interval in seconds, until it succeeds or the maximum number of attempts is reached.
    If the object still cannot be created or the title cannot be accessed after the maximum number
    of attempts, the last exception is raised.
    """
    last_exception = None
    for i in range(max_retries):
        try:
            yt = YouTube(url)
            title = yt.title  # Access the title of the YouTube object.
            return yt  # Return the YouTube object if successful.
        except Exception as err:
            last_exception = err  # Keep track of the last exception raised.
            print(
                f"Failed to create YouTube object or access title. Retrying... ({i+1}/{max_retries})"
            )
            # Wait for the specified interval before retrying.
            time.sleep(interval_secs)

    # If the YouTube object still cannot be created or the title cannot be accessed after the maximum number of attempts, raise the last exception.
    raise last_exception


if __name__ == "__main__":
    playlist = Playlist(playlist_url)
    playlist_video_urls = playlist.video_urls

    print(
        f"There are total {len(playlist_video_urls)} videos in the playlist.")

    video_data = []
    for video in tqdm(playlist_video_urls, total=len(playlist_video_urls)):
        try:
            curr_video_data = {}
            yt = retry_access_yt_object(video, max_retries=25, interval_secs=2)
            curr_video_data["title"] = yt.title
            curr_video_data["url"] = video
            curr_video_data["duration"] = yt.length
            curr_video_data["publish_date"] = yt.publish_date.strftime(
                "%Y-%m-%d")
            loader = YoutubeLoader.from_youtube_url(video)
            transcript = loader.load()[0].page_content
            transcript = " ".join(transcript.split())
            curr_video_data["transcript"] = transcript
            curr_video_data["total_words"] = len(transcript.split())
            video_data.append(curr_video_data)
        except:
            print(f"Failed to scrape {video}")

    print(f"Total podcast episodes scraped: {len(video_data)}")

    # save the scraped data to a csv file
    df = pd.DataFrame(video_data)
    data_path = Path(root_data_dir, "yt_podcast_transcript.csv")
    df.to_csv(data_path, index=False)
