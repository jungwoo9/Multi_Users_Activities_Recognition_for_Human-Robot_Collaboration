def validate_participant_ids(path_directory):
    """
    Validates the correctness of research participant IDs (object IDs) in pair data markers (skeleton directories).
    IDs must be either 100 or 200; otherwise, it indicates a reassignment error.
    """
    for directory in sorted(path_directory.iterdir()):
        if directory.is_dir():
            if str(directory).split("_")[-1] == "skeleton":
                for db in directory.iterdir():
                    if db.is_file():
                        _, _, _, data = extract_data_from_db(db)
                        for i in data:
                            if i[-1] not in [100, 200]:
                                print("Include new id(reassigend):", i[-1])

def generate_paired_data(skeleton, label):
    """
    pair skeleton and map label
    """

    temp_id = -100
    new_skeleton = []
    new_label = []

    for i in range(len(skeleton)):
        id = skeleton[i][-1]
        s = skeleton[i][:-1]
        l = label[i]

        if id != temp_id:
            temp_id = id
            new_skeleton.append(s)
            new_label.append(l)
        
        new_skeleton[-1] = s
        new_label[-1] = l

    print(new_skeleton)
    print(new_label)