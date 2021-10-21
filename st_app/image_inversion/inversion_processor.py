from st_app.image_inversion.face_alignment import LandmarksDetector, FaceAlignmentProcessor


class InversionPreprocessor():
    def __init__(self, device):
        self.landmarks_detector = LandmarksDetector(device=device)
        self.face_aligment = FaceAlignmentProcessor(256)

    def __call__(self, image, faces_landmarks=None, multifaces=True):
        # (Optionnal faces landmarks extraction)
        if faces_landmarks is None:
            faces_landmarks = self.landmarks_detector.get_all_face_landmarks(image)

        print(faces_landmarks[0])
        # Case: No face detected
        if len(faces_landmarks) == 0:
            print('No face detected in image.')
            return [], []

        # face alignement
        aligned_images, inverse_transformations = self.face_aligment.get_all_aligned_images(image, faces_landmarks)
        if not multifaces:
            aligned_images, inverse_transformations = [aligned_images[0]], [inverse_transformations[0]]

        return aligned_images, inverse_transformations