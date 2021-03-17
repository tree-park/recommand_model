import requests
import shutil


def get_image(filepath, image_url):
    filename = image_url[0]

    r = requests.get(image_url[1], stream=True)
    if r.status_code == 200:
        r.raw.decode_content = True
        with open(filepath + filename + '.jpg', 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ', filepath + filename)
    else:
        print('Image Couldn\'t be retreived')
