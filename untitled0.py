import csv
import cv2
import numpy as np
import os
import sys
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

images_in_folder = 'C:\PoseItPerfect\PoseItPerfect\PoseItPerfect\posify\public\poses_images_in'

images_out_folder = 'fitness_poses_images_out_basic'

csv_out_path = 'poses_csvs_out_basic3.csv'

with open(csv_out_path, 'w', newline='') as csv_out_file:
  csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
  csv_out_writer.writerow(['ImageName', 'ImageFolder','Width ', 'Height', '0x', '0y', '0z', '1x', '1y', '1z', '2x', '2y', '2z', '3x', '3y', '3z',
                             '4x', '4y', '4z', '5x', '5y', '5z', '6x', '6y', '6z', '7x', '7y', '7z'
                             , '8x', '8y', '8z', '9x', '9y', '9z', '10x', '10y', '10z', '11x', '11y', '11z'
                             , '12x', '12y', '12z', '13x', '13y', '13z', '14x', '14y', '14z', '15x', '15y', '15z'
                             , '16x', '16y', '16z', '17x', '17y', '17z', '18x', '18y', '18z', '19x', '19y', '19z'
                             , '20x', '20y', '20z', '21x', '21y', '21z', '22x', '22y', '22z', '23x', '23y', '23z'
                             , '24x', '24y', '24z', '25x', '25y', '25z', '26x', '26y', '26z', '27x', '27y', '27z'
                             , '28x', '28y', '28z', '29x', '29y', '29z', '30x', '30y', '30z', '31x', '31y', '31z'
                             , '32x', '32y', '32z'])


  # Folder names are used as pose class names.
  pose_class_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])

  for pose_class_name in pose_class_names:
    print('Bootstrapping ', pose_class_name, file=sys.stderr)

    if not os.path.exists(os.path.join(images_out_folder, pose_class_name)):
      os.makedirs(os.path.join(images_out_folder, pose_class_name))

    image_names = sorted([
        n for n in os.listdir(os.path.join(images_in_folder, pose_class_name))
        if not n.startswith('.')])
    for image_name in tqdm.tqdm(image_names, position=0):
      # Load image.
      input_frame = cv2.imread(os.path.join(images_in_folder, pose_class_name, image_name))
      input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

      # Initialize fresh pose tracker and run it.
      with mp_pose.Pose() as pose_tracker:
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

      # Save image with pose prediction (if pose was detected).
      output_frame = input_frame.copy()
      if pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=output_frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)
      output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

      # Save landmarks.
      if pose_landmarks is not None:
        # Check the number of landmarks and take pose landmarks.
        assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(pose_landmarks.landmark))
        imageHeight, imageWidth, imageChannel = input_frame.shape

        # Map pose landmarks from [0, 1] range to absolute coordinates to get
        # correct aspect ratio.
        frame_height, frame_width = output_frame.shape[:2]
        a = []
        # pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
        for lmk in pose_landmarks.landmark:
          if(lmk.y * frame_height <= imageHeight and lmk.x * frame_width<= imageWidth):
            a.append([lmk.x, lmk.y, lmk.z])
          else:
            a.append([0,0,0])
        pose_landmarks = a
        pose_landmarks *= np.array([frame_width, frame_height, frame_width])

        # Write pose sample to CSV.
        pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str).tolist()
        csv_out_writer.writerow([image_name, pose_class_name, imageWidth, imageHeight] + pose_landmarks)

# files.download(csv_out_path)

