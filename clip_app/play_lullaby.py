# When called will activate a lullaby / song / melody

from playsound import playsound
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MP3_FILE = os.path.join(CURRENT_DIRECTORY, "..", "resources", "brahms-lullaby.mp3")
def play_mp3(mp3_file_path=DEFAULT_MP3_FILE):
    try:
        # Play the MP3 file
        print("Playing the MP3 file...")
        playsound(mp3_file_path)
    except Exception as e:
        print(f"An error occurred while playing the file: {e}")

if __name__ == "__main__":
    play_mp3()
