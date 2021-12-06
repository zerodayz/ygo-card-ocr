# This is an Yu-gi-Oh! card-ocr script

import cv2
import os
import pytesseract
import requests
from lxml import html


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 88, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def read_card_name(image_file):
    card_names = []

    # Reading image
    img = cv2.imread(image_file)

    # Cropping each row of cards
    first_row = img[170:310]
    second_row = img[1470:1610]
    third_row = img[2770:2910]

    # Feel free to adjust the rows if needed
    card1 = first_row[:, 300:1000]
    card2 = first_row[:, 1200:1900]
    card3 = first_row[:, 2100:2800]

    card4 = second_row[:, 300:1000]
    card5 = second_row[:, 1150:1900]
    card6 = second_row[:, 2000:2750]

    card7 = third_row[:, 300:1000]
    card8 = third_row[:, 1150:1900]
    card9 = third_row[:, 2000:2750]

    # PyTesseract config
    custom_config = r'--psm 7 -c tessedit_char_whitelist=' \
                    r'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

    for image in (card1, card2, card3, card4, card5,
                  card6, card7, card8, card9):

        # Convert image to Grayscale
        card_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # alpha is contrast control, beta is brightness
        new_image = cv2.convertScaleAbs(card_gray, alpha=1, beta=0)
        new_image = image_smoothening(new_image)

        # Detect text from image
        texts = pytesseract.image_to_string(image, config=custom_config)
        query = texts.replace("\n\f", "")
        url = 'https://yugiohprices.com/search_card?search_text=' + query
        r = requests.get(url)
        tree = html.fromstring(r.content)
        card_name = tree.xpath('//h1[@id="item_name"]/text()')
        if len(card_name) == 0:
            card_names.append("Unknown")
        else:
            card_names.append(card_name[0])

    print(card_names)


if __name__ == '__main__':
    directory = '/Users/rrobin/Downloads/'
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg"):
            print(os.path.join(directory, filename))
            read_card_name(os.path.join(directory, filename))
            continue
        else:
            continue
