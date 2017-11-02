import cv2

''' Returns generator of patches from specified image and its center points '''


class SlidesProducer:
    def __init__(self, img_path, window_size=(64, 64), step_size=1):
        self.image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.window_size = window_size
        self.step_size = step_size

    def get_image(self):
        return self.image

    def sliding_window(self):
        # slide a window across the image
        for x in range(self.step_size, self.image.shape[1] - self.step_size, self.step_size):
            for y in range(self.step_size, self.image.shape[0] - self.step_size, self.step_size):
                # yield the current window
                yield (x + int(self.window_size[0]/2), y + int(self.window_size[0]/2), self.image[y:y + self.window_size[1], x:x + self.window_size[0]])

    def __call__(self):
        # get the cropped list of patches and its center points
        return self.sliding_window()
