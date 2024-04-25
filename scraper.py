import re
import os
import io
import cv2
import numpy as np
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from PIL import Image
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def scrape_myntra(product, num_pages):
    product = product.replace(' ','-')
    driver = webdriver.Chrome()
    domain = 'https://www.myntra.com/'
    links = []

    # Scraping product links
    for page in tqdm(range(1, num_pages + 1), desc='Scraping Product Links', unit = 'links'):
        url = f'https://www.myntra.com/{product}?p={page}'
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        for item in soup.find_all('li', class_='product-base'):
            link = item.find('a', {'data-refreshpage': 'true'})
            if link:
                links.append(domain + link.get('href'))

    data = []
    # Scraping product details from each link
    for link in tqdm(links, desc='Scraping Product Details', unit = 'product'):
        driver.get(link)
        htmlu = driver.page_source
        soupu = BeautifulSoup(htmlu, 'html.parser')
        
        brand = soupu.find('h1', attrs={'class': 'pdp-title'}).text if soupu.find('h1', attrs={'class': 'pdp-title'}) else None
        name = soupu.find('h1', attrs={'class': 'pdp-name'}).text if soupu.find('h1', attrs={'class': 'pdp-name'}) else None
        description = soupu.find('p', attrs={'class': 'pdp-product-description-content'}).text if soupu.find('p', attrs={'class': 'pdp-product-description-content'}) else None
        x = soupu.find('div', attrs={'class': 'index-overallRating'})
        rating = x.text[:3] if x else None
        price = soupu.find('span', attrs={'class': 'pdp-price'}).text if soupu.find('span', attrs={'class': 'pdp-price'}) else None
        code = soupu.find('span', attrs={'class': 'supplier-styleId'}).text if soupu.find('span', attrs={'class': 'supplier-styleId'}) else None
        reviews = [i.text.strip() for i in soupu.findAll('div', attrs={'class': 'user-review-reviewTextWrapper'})] if soupu.findAll('div', attrs={'class': 'user-review-reviewTextWrapper'}) else None
        image = re.search(r'url\("(.*?)"\)', [div['style'] for div in soupu.findAll('div', attrs={'class':'image-grid-image'})][0]).group(1) if soupu.findAll('div', attrs={'class':'image-grid-image'}) else None

        data.append({
            'Brand': brand,
            'Name': name,
            'Description': description,
            'Rating': rating,
            'Price': price,
            'Code': code,
            'Reviews': reviews,
            'Image' : image
        })

    driver.quit()
    return pd.DataFrame(data)

def get_images(work_dir, link, name):
    re = requests.get(link)
    image_stream = io.BytesIO(re.content)
    image = Image.open(image_stream)
    image.save(work_dir + str(name) + '.jpg')

def save_image(work_dir, name, df):
    directory = work_dir + name + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    for i in trange(len(df), unit=' images', desc='Saving Images'):
        try:
            get_images(directory, df['Image'][i], df['Code'][i])
        except requests.exceptions.MissingSchema:
            pass

def predict(image_path,model1,model2):
    gender_dict = {
    '0': {'Gender': 'Female'},
    '1': {'Gender': 'Male'}}

    sleeve_dict = {
    '0': {'Sleeve': 'Full'},
    '1': {'Sleeve': 'Half'}}
    image = plt.imread(image_path)
    resized = cv2.resize(image, (512,512))
    test_image = resized.reshape((1,512,512,3))
    class1 = int(np.round(model1.predict(test_image),0)[0,0])
    class2 = int(np.round(model2.predict(test_image),0)[0,0])

    plt.imshow(image)
    plt.title(f'{gender_dict.get(str(class1)), sleeve_dict.get(str(class2))}')
    plt.show()