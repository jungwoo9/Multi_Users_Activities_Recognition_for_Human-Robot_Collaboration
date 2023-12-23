from util.zstd_decompressor import zstd_decompressor
from pathlib import Path
import os

zstd_file_lst = []
directory = Path("./paired_users")

# change file name from zst to zstd
rename = True
if rename:
    for d in sorted(directory.iterdir()):
        if d.is_dir():
            for f in d.iterdir():
                if f.name.split(".")[-1] == "zst":
                    if rename:
                        os.rename(f, f.with_suffix('.zstd'))
                        # print(f.with_suffix('.zstd'))

# get all the files
for d in sorted(directory.iterdir()):
    if d.is_dir():
        for f in d.iterdir():
            if f.name.split(".")[-1] == "zstd":
                zstd_file_lst.append(f)

# decompress and save files
path_save_dir = Path("./paired_users_decompressed")
for f in zstd_file_lst:
    path_tmp = path_save_dir / Path(*f.parts[1:2])
    path_tmp.mkdir(parents=True, exist_ok=True)
    zstd_decompressor(f, path_tmp / Path(".".join(f.name.split(".")[:-1])))
