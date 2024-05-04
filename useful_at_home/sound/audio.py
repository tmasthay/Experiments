import hydra
from omegaconf import DictConfig
from pydub import AudioSegment
from pydub.playback import play
import time
import os

@hydra.main(config_path="cfg", config_name="cfg")
def play_audio(cfg: DictConfig) -> None:
    # Load the audio file
    file_path = os.path.join(os.path.dirname(__file__), cfg.audio.file_path)
    audio = AudioSegment.from_file(os.path.abspath(file_path))
    
    # Calculate the total number of iterations to play
    max_iters = int(cfg.audio.total_time / cfg.audio.interval)
    
    # Segment to be played
    segment = audio[cfg.audio.start * 1000:cfg.audio.end * 1000]
    
    # Loop to play the audio segment at the specified intervals
    for _ in range(max_iters):
        play(segment)
        time.sleep(cfg.audio.interval)

if __name__ == "__main__":
    play_audio()
