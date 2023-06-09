import cv2
from PIL import Image, ImageDraw

def draw_image(image, box_pred, box_target, point_pred, point_target) -> Image:

    img = (image.cpu().permute(2, 1, 0).repeat(1,1,3).numpy()*255)
    img = (cv2.resize(img, dsize=[640,480])).astype("uint8")
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # draw target, prediction and distance
    w, h, c = img.shape
    x_1, y_1, x_2, y_2 = box_target.sum(0).sum(0)
    draw.rectangle([ y_1*(h), x_1*w, y_2*h, x_2*w], fill = None, outline ='blue', width=3) # draw the target in blue

    x_1, y_1, x_2, y_2 = box_pred.sum(0).sum(0)
    draw.rectangle([ y_1*h, x_1*w, y_2*h, x_2*w], fill = None, outline ='red', width=3) # draw the target in blue

    box = 2
    draw.ellipse((point_pred[0]-box, point_pred[1]-box, point_pred[0]+box, point_pred[1]+box), fill = 'green', outline ='blue') # draw the target in blue
    draw.ellipse((point_target[0]-box, point_target[1]-box, point_target[0]+box, point_target[1]+box), fill = 'black', outline ='blue') # draw the target in blue

    return pil_img
