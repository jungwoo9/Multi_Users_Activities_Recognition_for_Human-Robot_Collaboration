import zstandard

def zstd_decompressor(path_source, path_save):
    """
    decompress file in path_source.
    db3.zstd to db3
    Args:
        path_source: path to file to decompress
        path_save: path to save the decomrpessed file
    """
    with open(path_source, "rb") as f:
        decompressor = zstandard.ZstdDecompressor()

        with open(path_save, "wb") as output:
            output.write(decompressor.decompress(f.read()))

