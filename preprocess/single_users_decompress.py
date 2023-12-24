from util.zstd_decompressor import zstd_decompressor
from pathlib import Path

# get all the files which is type of 'zstd'
zstd_file_lst = []
directory = Path("./data/raw/single_users")
for d in directory.iterdir():
    if d.is_dir():
        for f in d.iterdir():
            if f.name.split(".")[-1] == "zstd":
                zstd_file_lst.append(f)        

# decompress and save files
path_save_dir = Path("./data/decompressed/single_users_decompressed")
for f in zstd_file_lst:
    path_tmp = path_save_dir / Path(*f.parts[1:2])
    path_tmp.mkdir(parents=True, exist_ok=True)
    zstd_decompressor(f, path_tmp / Path(".".join(f.name.split(".")[:-1])))