import numpy as np
import matplotlib.pyplot as plt
import cv2

class ShapeDrawer:
    @staticmethod
    def draw_circle():
        image = np.ones((200, 200, 3), dtype="uint8") * 255
        center = (100, 100)
        radius = 50
        color = (128, 0, 128)  # Mor renk
        cv2.circle(image, center, radius, color, -1)
        return image

    @staticmethod
    def draw_square():
        image = np.ones((200, 200, 3), dtype="uint8") * 255
        top_left = (75, 75)
        bottom_right = (125, 125)
        color = (128, 0, 128)  # Mor renk
        cv2.rectangle(image, top_left, bottom_right, color, -1)
        return image

class EdgeDetector:
    horizontal_filter = np.array([[-1, 1]])
    vertical_filter = np.array([[-1], [1]])

    @staticmethod
    def apply_filters(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        horizontal_derivative = cv2.filter2D(gray_image, -1, EdgeDetector.horizontal_filter)
        vertical_derivative = cv2.filter2D(gray_image, -1, EdgeDetector.vertical_filter)
        return horizontal_derivative + vertical_derivative

mavi_daire = ShapeDrawer.draw_circle()
mavi_kare = ShapeDrawer.draw_square()

daire_turev = EdgeDetector.apply_filters(mavi_daire)
kare_turev = EdgeDetector.apply_filters(mavi_kare)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(cv2.cvtColor(mavi_daire, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Orijinal Mor Daire')
axs[0, 0].axis('off')


axs[0, 1].imshow(daire_turev, cmap='gray')
axs[0, 1].set_title('Türev Uygulanmış Daire')
axs[0, 1].axis('off')


axs[1, 0].imshow(cv2.cvtColor(mavi_kare, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Orijinal Mor Kare')
axs[1, 0].axis('off')


axs[1, 1].imshow(kare_turev, cmap='gray')
axs[1, 1].set_title('Türev Uygulanmış Kare')
axs[1, 1].axis('off')

plt.show()
