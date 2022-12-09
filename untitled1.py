import pandas as pd
import csv

point_indices = {'nose': 0,
                 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
                 'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
                 'left_ear': 7, 'right_ear': 8,
                 'mouth_left': 9, 'mouth_right': 10,
                 'left_shoulder': 11, 'right_shoulder': 12,
                 'left_elbow': 13, 'right_elbow': 14,
                 'left_wrist': 15, 'right_wrist': 16,
                 'left_pinky_1': 17, 'right_pinky_1': 18,
                 'left_index_1': 19, 'right_index_1': 20,
                 'left_thumb_2': 21, 'right_thumb_2': 22,
                 'left_hip': 23, 'right_hip': 24,
                 'left_knee': 25, 'right_knee': 26,
                 'left_ankle': 27, 'right_ankle': 28,
                 'left_heel': 29, 'right_heel': 30,
                 'left_foot_index': 31, 'right_foot_index': 32}


a = pd.read_csv('poses_csvs_out_basic3.csv')


def getCoordinatesOfABodyPoint(lst, n, start):
    res = [lst[n * 3 + start], lst[n * 3 + 1 + start], lst[n * 3 + 2 + start]]
    return res


def distanceBetweenTwoPoints(lst1, lst2):
    if lst1 == [0, 0, 0] or lst2 == [0, 0, 0]:
        return 0
    x = pow((lst2[0] - lst1[0]), 2)
    y = pow((lst2[1] - lst1[1]), 2)
    z = pow((lst2[2] - lst1[2]), 2)
    return pow((x + y + z), 0.5)


def getAverage(d1, d2):
    if d2 == 0:
        return d1
    elif d1 == 0:
        return d2
    else:
        return (d1 + d2) / 2


def getCenterBetweenTwoPoints(lst1, lst2):
    if lst1 == [0, 0, 0] or lst2 == [0, 0, 0]:
        return [0, 0, 0]
    res = [(lst1[0] + lst2[0]) / 2, (lst1[1] + lst2[1]) / 2, (lst1[2] + lst2[2]) / 2]
    return res


def getInsights(lst, start=4):
    res = []
    shoulder_width = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_shoulder'], start),
                                              getCoordinatesOfABodyPoint(lst, point_indices['right_shoulder'], start))

    shoulder_center = getCenterBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_shoulder'], start),
                                                getCoordinatesOfABodyPoint(lst, point_indices['right_shoulder'], start))
    mouth_center = getCenterBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['mouth_right'], start),
                                             getCoordinatesOfABodyPoint(lst, point_indices['mouth_left'], start))
    neck_distance = getNeckDistance(mouth_center, shoulder_center)

    waist_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_hip'], start),
                                              getCoordinatesOfABodyPoint(lst, point_indices['right_hip'], start))

    waist_center = getCenterBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_hip'], start),
                                             getCoordinatesOfABodyPoint(lst, point_indices['right_hip'], start))
    shoulder_to_waist_distance = distanceBetweenTwoPoints(shoulder_center, waist_center)

    eye_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_eye'], start),
                                            getCoordinatesOfABodyPoint(lst, point_indices['right_eye'], start))

    eye_center = getCenterBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_eye'], start),
                                           getCoordinatesOfABodyPoint(lst, point_indices['right_eye'], start))
    ear_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_ear'], start),
                                            getCoordinatesOfABodyPoint(lst, point_indices['right_ear'], start))

    eye_to_nose_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['nose'], start), eye_center)

    eye_to_mouth_distance = distanceBetweenTwoPoints(eye_center, mouth_center)

    left_hip_to_knee_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_hip'], start),
                                                         getCoordinatesOfABodyPoint(lst, point_indices['left_knee'], start))
    right_hip_to_knee_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['right_hip'], start),
                                                          getCoordinatesOfABodyPoint(lst, point_indices['right_knee'], start))
    hip_to_knee_distance = getAverage(left_hip_to_knee_distance, right_hip_to_knee_distance)

    left_knee_to_foot_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['left_knee'], start),
                                                          getCoordinatesOfABodyPoint(lst, point_indices['left_ankle'], start))
    right_knee_to_foot_distance = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['right_knee'], start),
                                                           getCoordinatesOfABodyPoint(lst, point_indices['right_ankle'], start))
    knee_to_foot_distance = getAverage(left_knee_to_foot_distance, right_knee_to_foot_distance)

    left_shoulder_to_elbow_distance = distanceBetweenTwoPoints(
        getCoordinatesOfABodyPoint(lst, point_indices['left_shoulder'], start),
        getCoordinatesOfABodyPoint(lst, point_indices['left_elbow'], start))
    right_shoulder_to_elbow_distance = distanceBetweenTwoPoints(
        getCoordinatesOfABodyPoint(lst, point_indices['right_shoulder'], start),
        getCoordinatesOfABodyPoint(lst, point_indices['right_elbow'], start))
    shoulder_to_elbow_distance = getAverage(left_shoulder_to_elbow_distance, right_shoulder_to_elbow_distance)

    left_elbow_to_wrist_distance = distanceBetweenTwoPoints(
        getCoordinatesOfABodyPoint(lst, point_indices['left_elbow'], start),
        getCoordinatesOfABodyPoint(lst, point_indices['left_wrist'], start))
    right_elbow_to_wrist_distance = distanceBetweenTwoPoints(
        getCoordinatesOfABodyPoint(lst, point_indices['right_elbow'], start),
        getCoordinatesOfABodyPoint(lst, point_indices['right_wrist'], start))
    elbow_to_wrist_distance = getAverage(left_elbow_to_wrist_distance, right_elbow_to_wrist_distance)

    mouth_width = distanceBetweenTwoPoints(getCoordinatesOfABodyPoint(lst, point_indices['mouth_left'], start),
                                           getCoordinatesOfABodyPoint(lst, point_indices['mouth_right'], start))

    res.append(eye_distance)
    res.append(ear_distance)
    res.append(eye_to_nose_distance)
    res.append(eye_to_mouth_distance)
    res.append(mouth_width)
    res.append(shoulder_width)
    res.append(neck_distance)
    res.append(waist_distance)
    res.append(shoulder_to_waist_distance)
    res.append(hip_to_knee_distance)
    res.append(knee_to_foot_distance)
    res.append(shoulder_to_elbow_distance)
    res.append(elbow_to_wrist_distance)

    return res


def getNeckDistance(noseLst, shoulderLst):
    h = distanceBetweenTwoPoints(noseLst, shoulderLst)
    p = abs(noseLst[2] - shoulderLst[2])
    if h == 0:
        return 0
    return pow(pow(h, 2) - pow(p, 2), 0.5)


file_name = 'image_insights2.csv'
with open(file_name, 'w', newline='') as out_csv_file:
    csvwriter = csv.writer(out_csv_file)
    head = ['ImageName', 'ImageFolder', 'Width', 'Height',
            'EyeDistance', 'EarDistance', 'EyeToNoseDistance', 'EyeToMouthDistance',
            'MouthWidth', 'ShoulderWidth', 'NeckDistance', 'WaistDistance',
            'ShoulderToWaistDistance', 'HipToKneeDistance', 'KneeToFootDistance', 'ShoulderToElbowDistance',
            'ElbowToWristDistance']
    csvwriter.writerow(head)
    rows, columns = a.shape
    for i in range(0, rows):
        lst = []
        csvRow = []
        b = a.iloc[i]
        for j in range(0, len(b)):
            lst.append(b[j])
        csvRow.append(b[0])
        csvRow.append(b[1])
        csvRow.append(b[2])
        csvRow.append(b[3])
        ins = getInsights(lst)
        csvRow.extend(ins)
        csvwriter.writerow(csvRow)

