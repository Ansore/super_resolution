from bs4 import BeautifulSoup
import requests
import uuid

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

# chrome_options = Options()
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--disable-gpu')

# browser = webdriver.Chrome(chrome_options=chrome_options)

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
}


def download_png(img_url):
    r = requests.get(img_url, stream=True)
    if r.status_code == 200:
        open('./images/'+str(uuid.uuid1())+'.png', 'wb').write(r.content)
        print(img_url+"----------done")
    else:
        print(img_url + "----------failed")
    del r


def get_url_list(host_html_url_list):
    result = []
    r = requests.get(host_html_url_list, headers=headers)
    html_soup = BeautifulSoup(r.text, 'html.parser')
    for i in html_soup.find('ul', id='post-list-posts').find_all('li'):
        # print(i.find('span', class_='plid').string.split(' ')[1])
        result.append(i.find('span', class_='plid').string.split(' ')[1])
    return result


def get_image_url(image_page_url):
    r2 = requests.get(image_page_url, headers=headers)
    html_soup2 = BeautifulSoup(r2.text, 'html.parser')
    if html_soup2.find('a', id='png') is None:
        return None
    return html_soup2.find('a', id='png').get('href')


if __name__ == "__main__":
    host_url = "http://konachan.net/post?page="
    index = 1
    for i in range(2, 100):
        host_url_temp = host_url + str(i)
        url_list = get_url_list(host_url_temp)
        for url in url_list:
            image_url = get_image_url(url)
            if image_url is None:
                print("---")
            else:
                download_png(image_url)
                print("finish " + str(index))
                index += 1
