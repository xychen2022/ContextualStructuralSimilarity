import numpy as np
import SimpleITK as sitk

subject_list = list(range(1, 21))

pancreas = []
for idx in range(len(subject_list)):
    subject_id = subject_list[idx]

    image = sitk.ReadImage('./segmentation/subject{0}/pred{0:04d}.nii.gz'.format(subject_id))
    image = sitk.GetArrayFromImage(image)

    label1 = np.zeros_like(image, dtype=np.float32)
    label1[np.where(image == 1)] = 1.

    
    groundtruth = sitk.ReadImage('/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label_uniform_v2/label{:04d}.nii.gz'.format(subject_id))
    groundtruth = sitk.GetArrayFromImage(groundtruth)

    groundtruth1 = np.zeros_like(groundtruth, dtype=np.float32)
    groundtruth1[np.where(groundtruth == 1)] = 1.
    groundtruth1 = groundtruth1.astype(np.bool)

    intersection = np.logical_and(label1, groundtruth1)
    im_sum = label1.sum() + groundtruth1.sum()

    print('DSC of subject{0} is: {1}'.format(subject_id, 2. * intersection.sum() / im_sum))
    pancreas.append(2. * intersection.sum() / im_sum)
    print('\n')

print('pancreas: ', pancreas, '\nmean: ', np.mean(pancreas), ' std: ', np.std(pancreas))
