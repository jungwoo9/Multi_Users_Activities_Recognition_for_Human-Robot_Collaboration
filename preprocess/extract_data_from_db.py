from rclpy import serialization
import sqlite3
from rosidl_runtime_py.utilities import get_message
from pathlib import Path

def extract_data_from_db(path_db):
    """
    Extracts id(row index in database), topic id, timestamp, and data from a database file.
    The data can be either label or skeleton. 
    For pair data, participant id is included in the skeleton.
    
    Uses a predefined joint index to extract specific 11 joints from the skeleton data.
    """

    # select joints for extraction
    azure_joint_index = {"PELVIS":0, "SPINE_NAVAL":1, "SPINE_CHEST":2, "NECK":3, "CLAVICLE_LEFT":4, "SHOULDER_LEFT":5, "ELBOW_LEFT":6, "WRIST_LEFT":7, "HAND_LEFT":8, "HANDTIP_LEFT":9, "THUMB_LEFT":10, "CLAVICLE_RIGHT":11, "SHOULDER_RIGHT":12, "ELBOW_RIGHT":13, "WRIST_RIGHT":14, "HAND_RIGHT":15, "HANDTIP_RIGHT":16, "THUMB_RIGHT":17, "HIP_LEFT":18, "KNEE_LEFT":19, "ANKLE_LEFT":20, "FOOT_LEFT":21, "HIP_RIGHT":22, "KNEE_RIGHT":23, "ANKLE_RIGHT":24, "FOOT_RIGHT":25, "HEAD":26, "NOSE":27, "EYE_LEFT":28, "EAR_LEFT":29, "EYE_RIGHT":30, "EAR_RIGHT":31}

    joints = ["PELVIS", "SPINE_NAVAL", "SPINE_CHEST", "NECK", "HEAD", "SHOULDER_RIGHT", "SHOULDER_LEFT", "ELBOW_RIGHT",  "ELBOW_LEFT", "HAND_RIGHT", "HAND_LEFT"]
    Index = sorted([azure_joint_index[i] for i in joints])
    
    # connect database
    conn = sqlite3.connect(path_db)
    cursor = conn.cursor()

    # extract data from topics table
    query = f"SELECT * FROM topics;"
    cursor.execute(query)
    db_lst = cursor.fetchall()

    # get message type
    message_type = get_message(db_lst[0][2])

    # extract data from messages table
    query = f"SELECT * FROM messages;"
    cursor.execute(query)
    db_lst = cursor.fetchall()

    # disconnect database
    cursor.close()
    conn.close()

    id = []
    topic_id = []
    time_stamp = []
    data = []

    # process label data
    if "label" in str(path_db):
        for row in db_lst:
            id.append(row[0])
            topic_id.append(row[1])
            time_stamp.append(row[2])
            msg = serialization.deserialize_message(row[3], message_type)

            data.append(msg.data)

    # process skeleton
    else:
        for row in db_lst:
            id.append(row[0])
            topic_id.append(row[1])
            time_stamp.append(row[2])
            marker_array = serialization.deserialize_message(row[3], message_type)

            # skip when the marker_array is empty
            if marker_array.markers == []:
                continue
            
            # get 11 joints
            skeleton = [marker_array.markers[i].pose.position for i in Index]
            
            # skip when the marker does not include all joints we needed
            if len(skeleton) != 11:
                continue

            data.append(skeleton)

    return id, topic_id, time_stamp, data