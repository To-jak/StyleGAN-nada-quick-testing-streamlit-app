import math
import numpy as np
import scipy.ndimage
import torch
from PIL import Image
import face_alignment


class LandmarksDetector():
    def __init__(self, device):
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)


    def get_all_face_landmarks(self, img):
        """
        Return a list of landmarks groups extraced from each face in the image.

        Outputs:
        - faces_landmarks_dict_list: list of {'eye_left': [x, y], 'eye_right': [x, y], 'mouth_avg': [x, y]}
        """
        ratio = 640 / max(img.size) # Resizing image for faster face detection
        img_resized = img.resize((int(img.width*ratio), int(img.height*ratio)))
        img_resized = np.array(img_resized)
        with torch.no_grad():
            faces_landmarks = self.model.get_landmarks(img_resized)
            faces_landmarks = [landmarks[:] * 1/ratio for landmarks in faces_landmarks]

        faces_landmarks_dict_list = []
        for face_landmarks in faces_landmarks:
            lm_eye_left      = face_landmarks[36 : 42]
            lm_eye_right     = face_landmarks[42 : 48]
            lm_mouth_outer   = face_landmarks[48 : 60]

            eye_left     = np.mean(lm_eye_left, axis=0)
            eye_right    = np.mean(lm_eye_right, axis=0)

            mouth_left   = lm_mouth_outer[0]
            mouth_right  = lm_mouth_outer[6]
            mouth_avg    = (mouth_left + mouth_right) * 0.5

            faces_landmarks_dict_list.append({'eye_left': eye_left,
                                              'eye_right':eye_right,
                                              'mouth_avg': mouth_avg})
        return faces_landmarks_dict_list


class FaceAlignmentProcessor():
    def __init__(self, output_size):
        self.output_size = output_size


    def _get_inverse_transforms_from_face_quad(self, quad):
        quad_center = [(quad[0][0]+quad[2][0])/2, (quad[0][1]+quad[2][1])/2]
        quad_midlow = [(quad[1][0]+quad[2][0])/2, (quad[1][1]+quad[2][1])/2]
        center_to_midlow = np.array(quad_midlow) - np.array(quad_center)
        deg = math.atan2(*center_to_midlow) * (180 / math.pi)
        left_border = quad[1] - quad[0]
        height = np.linalg.norm(left_border)
        scale = height / 256

        transforms = {
            'rotation': deg,
            'scale': scale,
            'center': quad_center}
        return transforms


    def _align_in_the_wild_image(self, np_img, face_landmarks):
        eye_left     = face_landmarks['eye_left']
        eye_right    = face_landmarks['eye_right']
        mouth_avg    = face_landmarks['mouth_avg']

        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8) # This results in larger crops then the original FFHQ. For the original crops, replace 2.2 with 1.8
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Inverse geometric transformations for Collage
        inverse_transformations = self._get_inverse_transforms_from_face_quad(quad)

        # Load in-the-wild image.
        img = Image.fromarray(np_img)

        output_size = self.output_size
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Algined crop from face quad and resizing.
        img = img.transform((output_size, output_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        return img, inverse_transformations


    def get_all_aligned_images(self, image, faces_landmarks):
        """
        Return the list of image of each aligned face given the landmarks.
        Also returns the list of alignment inverse transformations, in order to replace each face in the original image for example.

        arguments:
        - image: input image.
        - faces_landmarks: list of {'eye_left': [x, y], 'eye_right': [x, y], 'mouth_avg': [x, y]}.

        outputs:
        - aligned_images: list of images (one for each face).
        - inverse_transformations: list of  {'rotation': deg, 'scale': scale, 'center': [x, y]}
        """
        np_img = np.array(image)
        aligned_images = []
        inverse_transformations = []
        for face_landmarks in faces_landmarks:
            aligned_img, inverse_transformation = self._align_in_the_wild_image(np_img, face_landmarks)
            inverse_transformations.append(inverse_transformation)
            aligned_images.append(aligned_img)
        return aligned_images, inverse_transformations