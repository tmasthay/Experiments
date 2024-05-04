import hydra
from omegaconf import DictConfig
from pydub import AudioSegment
from pydub.playback import play
import time
import os

@hydra.main(config_path="cfg", config_name="cfg")
def play_audio(cfg: DictConfig) -> None:
    # Load the audio file
    file_path = os.path.join(os.path.dirname(__file__), cfg.file_path)
    audio = AudioSegment.from_file(os.path.abspath(file_path))
    
    # Calculate the total number of iterations to play
    max_iters = int(cfg.total_time / cfg.interval)
    
    # Segment to be played
    if type(cfg.end) == str and cfg.start.lower()[0] in ['a']:
        start = 0
        end = len(audio)
    else:
        start = cfg.start * 1000
        end = cfg.end * 1000
    segment = audio[start:end]
    
    # Loop to play the audio segment at the specified intervals
    for _ in range(max_iters):
        play(segment)
        time.sleep(cfg.interval)

if __name__ == "__main__":
    play_audio()
