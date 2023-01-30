import glob
import json

data_root = "data"


transcriptions = glob.glob(f"{data_root}/*/transcriptions.txt")
spk2id = {}
spk_id = 0
ms_transcriptions = open(f'{data_root}/transcriptions.txt', "w")
ms_train_set = open(f'{data_root}/train.list', "w")
ms_test_set = open(f'{data_root}/test.list', "w")
for transcription in transcriptions:
    spk = transcription.split("/")[-2]
    spk2id[spk] = spk_id
    spk_id += 1
    for line in open(transcription).readlines():
        ms_transcriptions.write(f"{spk}/{line}")
    for line in open(transcription.replace("transcriptions.txt", "train.list")):
        ms_train_set.write(f"{spk}/{line}")
    for line in open(transcription.replace("transcriptions.txt", "test.list")):
        ms_test_set.write(f"{spk}/{line}")

ms_transcriptions.close()
ms_train_set.close()
ms_test_set.close()
print("请手动将说话人与id的映射粘贴至config文件中")
print(json.dumps(spk2id))