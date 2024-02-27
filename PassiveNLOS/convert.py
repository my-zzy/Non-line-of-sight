from PIL import Image
import numpy as np
import scipy.io
from scipy.io import savemat

for i in range(248):

    image = Image.open('./dataset/pic/{}.tif'.format(i))


    # print("图像格式:", image.format)
    # print("图像大小:", image.size)
    # print("图像模式:", image.mode)

    # image.show()

    width, height = image.size

    target_width = 400
    target_height = 400

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    cropped_image = image.crop((left, top, right, bottom))
    # print(cropped_image.size)

    numpy_array = np.array(cropped_image, dtype=np.uint8)
    # print(numpy_array.shape)

    label = Image.open('./dataset/mnist-1000/{}.bmp'.format(i+1)).convert('L')

    # print(label.size)

    # if image.mode != 'RGB':
    #     image = image.convert('RGB')
    # red, green, blue = image.split()
    # green_data = green.getdata()

    # print(green_data.size)

    width, height = label.size


    target_width = 28
    target_height = 28

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    cropped_lb = label.crop((left, top, right, bottom))
    # print(cropped_lb.size)

    lb = np.array(cropped_lb, dtype=np.uint8)
    # print(lb.shape)
    
    # dat = scipy.io.loadmat('1.mat')
    # matrix = dat['label']
    # lb = np.array(matrix)
    # print(lb.shape)

    data = {'G': numpy_array, 'label': lb}
    savemat('./dataset/mat/{}.mat'.format(i), data)