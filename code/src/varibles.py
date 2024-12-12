import os
import requests
from typing import Union, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImgPreprocessor:
    def __init__(self, resize_factor=2, adaptive_threshold=False, deskew=False, denoise=True, contrast=True):
        self.resize_factor = resize_factor
        self.adaptive_threshold = adaptive_threshold
        self.deskew = deskew
        self.denoise = denoise
        self.contrast = contrast

    def preprocess(self, image_path, show_result=False):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        p_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if show_result:
            self.show_image("Grayscale", p_img)

        p_img = self.resize_image(p_img, show_result=show_result)

        if self.contrast:
            p_img = self.increase_contrast(p_img, show_result=show_result)

        if self.denoise:
            p_img = self.denoise_image(p_img, show_result=show_result)
            
        if self.adaptive_threshold:
            p_img = self.apply_threshold(p_img, show_result=show_result)

        if self.deskew:
            p_img = self.deskew_image(p_img, show_result=show_result)

        return p_img

    def resize_image(self, image, show_result=False):
        height, width = image.shape
        new_size = (int(width * self.resize_factor), int(height * self.resize_factor))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        if show_result:
            self.show_image("Resized", resized)
        return resized

    def denoise_image(self, image, show_result=False):
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        if show_result:
            self.show_image("Denoised", denoised)
        return denoised

    def increase_contrast(self, image, show_result=False):
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))
        contrasted = clahe.apply(image)
        contrasted = cv2.equalizeHist(contrasted)
        if show_result:
            self.show_image("Contrast Increased", contrasted)
        return contrasted

    def apply_threshold(self, image, show_result=False):
        thresholded = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        if show_result:
            self.show_image("Thresholded", thresholded)
        return thresholded

    def deskew_image(self, image, show_result=False):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if show_result:
            self.show_image("Deskewed", deskewed)
        return deskewed

    def show_image(self, title, image):
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis("off")
        plt.show()

    def save_preprocessed_image(self, image, output_path):
        cv2.imwrite(output_path, image)

class ImgDataMngr:
    """
    Manages image data for a given user, including downloading, preprocessing, and optional line 
    detection for OCR or analysis tasks. Can also handle cleanup of temporary image files.
    """

    def __init__(self, uid: str, group_ids: Union[int, List[int]],
                 entity_names: Union[str, List[str]],image_paths:Union[str, List[str]]=None, ImgLinks:Union[str, List[str]]=None, entity_values=None, DownloadPath=None, save_img=False):
        """
        Initializes image manager parameters and downloads/preprocesses images.

        Parameters:
        - uid (str): Unique identifier for the user or image group.
        - ImgLinks (str or list): URLs to images to download and process.
        - group_ids (int or list): Group IDs associated with each image.
        - entity_names (str or list): Entity names corresponding to each image.
        - entity_values (optional): Additional values linked to the images.
        - DownloadPath (str, optional): Directory path for saving downloaded images.
        - save_img (bool): If False, deletes downloaded images after processing.
        """
        self.uid = uid
        

        self.group_ids = [group_ids] if isinstance(group_ids, int) else group_ids
        self.entity_names = [entity_names] if isinstance(entity_names, str) else entity_names
        self.entity_values = entity_values if isinstance(entity_values, list) else [entity_values]
        self.preprocessor = ImgPreprocessor()
        self.save_img = save_img
        if ImgLinks:
            self.ImgLinks = [ImgLinks] if isinstance(ImgLinks, str) else ImgLinks
            self.DownloadPath = DownloadPath or 'TempImg'
            os.makedirs(self.DownloadPath, exist_ok=True)
            self.image_paths= []
            self.preProcessedImages = self.download_and_preprocess_images()
        else:
            self.image_paths=  [image_paths] if isinstance(image_paths, str) else image_paths
            self.preProcessedImages = []
            for image_path in self.image_paths:
                processed_image = self.preprocess_image(image_path)
                self.preProcessedImages.append(processed_image)
                if not self.save_img:
                    os.remove(image_path)
                
        self.ocr_texts = []

    def download_image(self, img_link: str, idx: int) -> str:
        """
        Downloads an image from a URL and saves it to the designated download path.

        Parameters:
        - img_link (str): URL of the image to download.
        - idx (int): Index of the image for identification.

        Returns:
        - str: File path of the downloaded image or an empty string if failed.
        """
        try:
            image_name = os.path.join(self.DownloadPath, f"{os.path.basename(img_link).split('.')[0]}-{self.uid}.jpg")
            response = requests.get(img_link, timeout=10)
            response.raise_for_status()
            with open(image_name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded image {idx + 1}: {image_name}")
            return image_name
        except requests.RequestException as e:
            print(f"Failed to download image {idx + 1} from {img_link}: {e}")
            return ""
        
    def preprocess_image(self,image_path,display_result = False):
        preprocessed_image = self.preprocessor.preprocess(image_path, show_result=False)
        if display_result:
            plt.figure(figsize=(8, 8))
            plt.imshow(preprocessed_image, cmap='gray')
            plt.title('Processed Image')
            plt.axis("off")
            plt.show()
        return preprocessed_image
    
    def download_and_preprocess_images(self) -> List[str]:
        """
        Downloads images concurrently, preprocesses each image, and optionally deletes 
        the raw downloaded files.

        Returns:
        - list of str: Paths to preprocessed images.
        """
        downloaded_images = []
        with ThreadPoolExecutor() as executor:
            future_to_link = {executor.submit(self.download_image, link, idx): idx for idx, link in enumerate(self.ImgLinks)}
            for future in as_completed(future_to_link):
                image_path = future.result()
                if image_path:
                    downloaded_images.append(image_path)
        
        # Preprocess each downloaded image and delete if save_img is False
        if self.save_img:
                self.image_paths = downloaded_images
        else: 
            self.image_paths = []
        processed_images = []
        for image_path in downloaded_images:
            processed_image = self.preprocess_image(image_path)
            processed_images.append(processed_image)
            if not self.save_img:
                os.remove(image_path)
        
        return processed_images

    # def detect_lines(self, display_images=True) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    #     """
    #     Detects lines in the first processed image using the Hough Line Transform.
        
    #     Parameters:
    #     - display_images (bool): If True, displays the detected lines.

    #     Returns:
    #     - list of tuple: Coordinates of detected lines.
    #     """
    #     if not self.image_filenames:
    #         print("No images available for line detection.")
    #         return []

    #     # Perform Canny edge detection on the first processed image
    #     edges = cv2.Canny(self.image_filenames[0], 50, 150)
        
    #     if display_images:
    #         plt.imshow(edges, cmap='gray')
    #         plt.title('Edge Detection')
    #         plt.axis('off')
    #         plt.show()

    #     # Apply Hough Transform to detect lines in the edge-detected image
    #     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    #     line_coordinates = []
        
    #     if lines is not None:
    #         for line in lines:
    #             x1, y1, x2, y2 = line[0]
    #             line_coordinates.append(((x1, y1), (x2, y2)))

    #     print(f"Number of lines detected: {len(line_coordinates)}")
    #     return line_coordinates

    # def __del__(self):
    #     """
    #     Destructor to delete downloaded images from disk if save_img is False.
    #     """
    #     if not self.save_img:
    #         for image_path in self.image_paths:
    #             if os.path.isfile(image_path):
    #                 try:
    #                     os.remove(image_path)
    #                     print(f"Deleted image: {image_path}")
    #                 except OSError as e:
    #                     print(f"Error deleting image {image_path}: {e}")


