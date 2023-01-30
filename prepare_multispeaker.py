import os
import shutil

for spk in os.listdir("data"):
    if os.path.isdir(f"data/{spk}"):
        if os.path.exists(f"data/{spk}/raw/wavs"):
            shutil.move(f"data/{spk}/raw/wavs", f"data/{spk}")
            shutil.move(f"data/{spk}/raw/transcriptions.txt", f"data/{spk}")
            shutil.rmtree(f"data/{spk}/raw")

