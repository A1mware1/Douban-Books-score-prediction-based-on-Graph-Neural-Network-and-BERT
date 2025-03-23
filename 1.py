import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import os
import re
import time
import json
# 替换为你的豆瓣登录后的 Cookie
COOKIES = {
    'll': '118124',
    'bid': 'r3zTnIxR7gY',
    '_pk_ref.100001.8cb4': '%5B%22%22%2C%22%22%2C1721411894%2C%22https%3A%2F%2Fbook.douban.com%2F%22%5D',
    '_pk_id.100001.8cb4': 'c00299452c59a741.1720765473.',
    '__utmv': '30149280.28201',
    'dbcl2': '"282019994:MnjJtCUFUf0"',
    '_ga': 'GA1.2.1499094153.1718068105',
    '_pk_ses.100001.8cb4': '1',
    'ct': 'y',
    'push_noty_num': '0',
    'push_doumail_num': '0',
    '__utma': '30149280.867390962.1716697279.1721408858.1721411895.23',
    '__utmz': '30149280.1721408858.22.19.utmcsr=baidu|utmccn=(organic)|utmcmd=organic',
    '__utmc': '30149280',
    '__utmt': '1',
    '__utmb': '30149280.25.10.1721411895'
}
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
]

# 手动添加代理IP
PROXIES = [
    'http://50.174.7.159:80',
    'http://189.202.188.149:80',
    'http://50.168.72.119:80',
    'http://50.217.226.47:80',
    'http://77.232.128.191:80',
    'http://198.199.86.11:8080',
    'http://50.168.72.122:80',
    'http://50.172.75.122:80',
    'http://50.168.72.116:80',
    'http://175.139.233.79:80',
    'http://34.91.114.10:8080',
    'http://50.223.239.161:80',
    'http://141.148.63.29:80',
    'http://85.214.195.118:80',
    'http://50.218.57.66:80',
    'http://143.244.130.209:80',
    'http://123.30.154.171:7777',
    'http://50.174.145.12:80',
    'http://207.148.71.74:80',
    'http://172.183.241.1:8080',
    'http://50.172.75.127:80',
    'http://139.162.78.109:8080',
    'http://50.175.212.76:80',
    'http://167.102.133.107:80',
    'http://50.144.189.54:80',
    'http://212.107.28.120:80',
    'http://51.75.33.162:80',
    'http://50.223.239.183:80',
    'http://50.223.239.165:80',
    'http://198.74.51.79:8888',
    'http://23.137.248.197:8888',
    'http://50.172.75.126:80',
    'http://50.174.145.13:80',
    'http://50.169.135.10:80',
    'http://50.231.110.26:80',
    'http://50.174.7.153:80',
    'http://50.174.145.9:80',
    'http://50.174.7.157:80',
    'http://50.168.72.118:80',
    'http://80.228.235.6:80',
    'http://202.61.204.51:80',
    'http://50.217.226.42:80',
    'http://50.175.212.79:80',
    'http://50.218.57.65:80',
    'http://49.249.155.3:80',
    'http://74.48.78.52:80',
    'http://50.172.75.123:80',
    'http://50.175.212.74:80',
    'http://34.81.160.132:80',
    'http://47.251.70.179:80',
    'http://50.230.222.202:80',
    'http://103.151.20.131:80',
    'http://50.218.57.67:80',
    'http://50.239.72.16:80',
    'http://50.218.57.68:80',
    'http://50.218.57.70:80',
    'http://161.35.70.249:3128',
    'http://50.217.226.41:80',
    'http://50.168.72.115:80',
    'http://34.23.45.223:80',
    'http://102.134.98.222:8081',
    'http://197.243.20.178:80',
    'http://20.205.61.143:80',
    'http://31.11.143.83:80',
    'http://50.175.212.77:80',
    'http://50.239.72.17:80',
    'http://50.174.145.14:80',
    'http://47.74.152.29:8888',
    'http://172.232.180.108:80',
    'http://50.217.226.43:80',
    'http://50.217.226.44:80',
    'http://50.218.57.69:80',
    'http://50.172.75.125:80',
    'http://24.205.201.186:80',
    'http://154.203.132.55:8090',
    'http://50.144.168.74:80',
    'http://213.218.228.253:80',
    'http://50.239.72.18:80',
    'http://172.173.132.85:80',
    'http://50.221.74.130:80',
    'http://195.23.57.78:80',
    'http://103.137.62.253:80',
    'http://50.223.239.166:80',
    'http://50.223.38.6:80',
    'http://50.174.7.152:80',
    'http://50.207.199.82:80',
    'http://50.223.242.97:80',
    'http://50.168.72.113:80',
    'http://50.202.75.26:80',
    'http://50.222.245.43:80',
    'http://50.144.166.226:80',
    'http://49.13.51.71:80',
    'http://50.218.204.106:80',
    'http://91.92.155.207:3128',
    'http://96.113.158.126:80',
    'http://12.186.205.120:80',
    'http://32.223.6.94:80',
    'http://149.56.148.20:80',
    'http://5.75.200.249:80',
    'http://50.223.239.168:80',
    'http://109.201.14.82:8080',
    'http://131.0.234.220:55555',
    'http://49.147.102.51:8081',
    'http://122.3.139.85:8181',
    'http://103.28.114.157:66',
    'http://103.89.233.226:83',
    'http://203.150.166.170:8080',
    'http://103.41.35.153:58080',
    'http://176.236.141.30:10001',
    'http://79.106.33.26:8079',
    'http://190.94.212.249:999',
    'http://157.20.233.213:8080',
    'http://38.188.249.77:8008',
    'http://94.75.76.3:8080',
    'http://31.24.44.115:3128',
    'http://50.218.57.74:80',
    'http://50.207.199.86:80',
    'http://119.9.77.49:8080',
    'http://50.207.199.87:80',
    'http://64.227.4.244:8888',
    'http://50.223.239.175:80',
    'http://159.65.77.168:8585',
    'http://13.81.217.201:80',
    'http://50.207.199.81:80',
    'http://211.128.96.206:80',
    'http://50.222.245.40:80',
    'http://8.219.97.248:80',
    'http://20.206.106.192:8123',
    'http://162.245.85.220:80',
    'http://50.223.246.226:80',
    'http://8.223.31.16:80',
    'http://50.217.226.40:80',
    'http://203.77.215.45:10000',
    'http://50.172.75.120:80',
    'http://66.191.31.158:80',
    'http://50.223.239.177:80',
    'http://89.116.23.45:80',
    'http://50.231.104.58:80',
    'http://123.205.24.244:8193',
    'http://195.252.236.91:80',
    'http://50.168.72.117:80',
    'http://50.174.145.8:80',
    'http://39.109.113.97:3128',
    'http://50.171.177.124:80',
    'http://181.41.194.186:80',
    'http://50.174.145.15:80',
    'http://50.217.226.45:80',
    'http://181.214.231.172:8080',
    'http://50.171.177.126:80',
    'http://50.174.7.156:80',
    'http://201.222.50.218:80',
    'http://50.207.199.80:80',
    'http://50.171.187.51:80',
    'http://51.210.214.28:80',
    'http://47.253.207.137:8080',
    'http://50.223.239.194:80',
    'http://139.60.209.2:80',
    'http://50.168.72.114:80',
    'http://50.218.224.35:80',
    'http://83.1.176.118:80',
    'http://50.174.7.162:80',
    'http://50.223.242.103:80',
    'http://41.207.187.178:80',
    'http://50.218.204.103:80',
    'http://50.218.204.99:80',
    'http://50.172.75.121:80',
    'http://50.174.7.154:80',
    'http://120.158.172.86:80',
    'http://50.171.122.30:80',
    'http://50.217.226.46:80',
    'http://5.35.92.156:80',
    'http://51.89.14.70:80',
    'http://198.49.68.80:80',
    'http://50.223.239.191:80',
    'http://50.146.203.174:80',
    'http://47.243.114.192:8180',
    'http://50.207.199.83:80',
    'http://68.178.203.69:8899',
    'http://50.231.172.74:80',
    'http://50.223.239.160:80',
    'http://149.56.18.62:8888',
    'http://212.92.148.162:8090',
    'http://68.185.57.66:80',
    'http://83.169.17.201:80',
    'http://50.172.39.98:80',
    'http://51.15.242.202:8888',
    'http://51.255.20.138:80',
    'http://50.207.199.84:80',
    'http://50.223.246.237:80',
    'http://50.223.239.185:80',
    'http://37.27.82.72:80',
    'http://65.109.199.3:80',
    'http://109.120.156.109:80',
    'http://198.44.255.3:80',
    'http://50.174.145.10:80',
    'http://47.251.43.115:33333',
    'http://188.132.209.245:80',
    'http://5.45.107.19:3128',
    'http://51.89.255.67:80',
    'http://47.88.31.196:8080',
    'http://127.0.0.7:80',
    'http://50.223.239.173:80',
    'http://103.184.56.125:8080',
    'http://45.225.120.36:40033',
    'http://181.39.24.155:999',
    'http://118.98.166.56:8080',
    'http://46.161.67.74:81',
    'http://103.160.69.97:8009',
    'http://49.13.27.107:8888',
    'http://35.185.196.38:3128',
    'http://47.252.29.28:11222',
    'http://216.137.184.253:80',
    'http://157.245.95.247:443',
    'http://200.174.198.86:8888',
    'http://119.47.90.38:1111',
    'http://69.171.76.99:8081',
    'http://65.109.194.23:80',
    'http://51.254.78.223:80',
    'http://20.204.214.79:3129',
    'http://20.44.188.17:3129',
    'http://20.204.212.45:3129',
    'http://198.16.63.10:80',
    'http://130.61.120.213:8888',
    'http://45.92.158.59:9098',
    'http://134.209.29.120:3128',
    'http://180.250.161.42:80',
    'http://154.203.132.49:8080',
    'http://93.177.67.178:80',
    'http://139.59.1.14:8080',
    'http://20.219.176.57:3129',
    'http://133.18.234.13:80',
    'http://188.40.59.208:3128',
    'http://20.44.189.184:3129',
    'http://116.203.28.43:80',
    'http://185.217.143.96:80',
    'http://103.105.196.30:80',
    'http://62.72.29.174:80',
    'http://114.156.77.107:8080',
    'http://20.127.221.223:80',
    'http://72.10.160.92:5635',
    'http://212.92.148.164:8090',
    'http://189.240.60.166:9090',
    'http://196.223.129.21:80',
    'http://189.240.60.164:9090',
    'http://167.99.236.14:80',
    'http://66.31.130.117:8080',
    'http://209.126.6.159:80',
    'http://186.121.214.210:32650',
    'http://202.51.121.59:8080',
    'http://14.170.154.193:19132',
    'http://50.222.245.47:80',
    'http://50.171.187.50:80',
    'http://67.43.228.250:26991',
    'http://72.10.160.170:2657',
    'http://190.103.177.131:80',
    'http://50.175.212.72:80',
    'http://50.122.86.118:80',
    'http://50.172.75.124:80',
    'http://20.24.43.214:80',
    'http://50.174.7.158:80',
    'http://50.175.212.66:80',
    'http://50.174.145.11:80',
    'http://85.8.68.2:80',
    'http://84.39.112.144:3128',
    'http://50.222.245.42:80',
    'http://189.240.60.171:9090',
    'http://50.221.230.186:80',
    'http://72.10.160.91:8167',
    'http://50.220.168.134:80',
    'http://50.223.239.167:80',
    'http://50.168.7.250:80',
    'http://50.168.72.112:80',
    'http://50.218.57.64:80',
    'http://50.222.245.41:80',
    'http://50.223.242.100:80',
    'http://50.239.72.19:80',
    'http://213.33.126.130:80',
    'http://80.120.49.242:80',
    'http://167.102.133.99:80',
    'http://50.218.57.71:80',
    'http://213.33.2.28:80',
    'http://103.247.21.234:8080',
    'http://124.105.180.29:8082',
    'http://38.156.73.56:8080',
    'http://103.69.20.28:58080',
    'http://103.154.230.58:8080',
    'http://103.155.62.158:8080',
    'http://157.100.60.170:999',
    'http://103.118.46.77:32650',
    'http://181.115.93.77:999',
    'http://41.139.169.99:8083',
    'http://189.127.190.109:8080',
    'http://79.174.12.190:80',
    'http://68.183.143.134:80',
    'http://217.182.210.152:80',
    'http://178.128.200.87:80',
    'http://47.56.110.204:8989',
    'http://0.0.0.0:80',
    'http://170.84.48.222:8080',
    'http://20.204.212.76:3129'
]

def get_all_tags(url, headers):
    try:
        response = requests.get(url, headers=headers, cookies=COOKIES)
        response.raise_for_status()
        print(f"Request to {url} successful with status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Request to {url} failed: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    tags = []
    category_sections = soup.select('div.article div')
    current_category = None

    for section in category_sections:
        category_name_tag = section.find('h2')
        if category_name_tag:
            current_category = category_name_tag.get_text(strip=True)
        tag_elements = section.select('table.tagCol a')
        for tag_element in tag_elements:
            tag_url = "https://book.douban.com" + tag_element['href']
            tag_name = tag_element.get_text(strip=True)
            tags.append((current_category, tag_name, tag_url))

    return tags

def get_user_info(user_url, headers):
    try:
        response = requests.get(user_url, headers=headers, cookies=COOKIES, proxies={"http": random.choice(PROXIES)})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        user_info_section = soup.find('div', class_='user-info')
        user_id = None
        user_join_date = None
        user_ip_location = None

        if user_info_section:
            pl_elements = user_info_section.find_all('div', class_='pl')
            if len(pl_elements) > 0:
                user_id_and_date = pl_elements[0].get_text(strip=True)
                user_id_match = re.search(r'\d+', user_id_and_date)
                join_date_match = re.search(r'\d{4}-\d{2}-\d{2}', user_id_and_date)
                if user_id_match:
                    user_id = user_id_match.group(0)
                if join_date_match:
                    user_join_date = join_date_match.group(0)
            ip_location_element = user_info_section.find('span', class_='ip-location')
            if ip_location_element:
                user_ip_location = ip_location_element.get_text(strip=True)

        user_info = {
            'User ID': user_id,
            'Join Date': user_join_date,
            'IP Location': user_ip_location
        }
        return user_info
    except Exception as e:
        print(f"Error occurred while fetching user info for {user_url}: {e}")
        return None

def get_all_comments(book_id, headers, max_comments=20):
    comments = []
    page = 0
    while len(comments) < max_comments:
        comments_url = f"https://book.douban.com/subject/{book_id}/comments/?start={page * 20}"
        headers['User-Agent'] = random.choice(USER_AGENTS)
        try:
            response = requests.get(comments_url, headers=headers, cookies=COOKIES, proxies={"http": random.choice(PROXIES)})
            response.raise_for_status()
            time.sleep(random.uniform(1, 3))  # 添加延迟
        except requests.RequestException as e:
            print(f"Request to {comments_url} failed: {e}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        comment_elements = soup.find_all('li', class_='comment-item')
        if not comment_elements:
            print(f"No comments found for book ID: {book_id}")
            break

        # 检查是否存在空评论
        blank_tip = soup.find('p', class_='blank-tip')
        if blank_tip and '还没人写过短评呢' in blank_tip.get_text():
            print(f"No valid comments for book ID: {book_id}")
            break

        for comment_element in comment_elements:
            if len(comments) >= max_comments:
                break
            try:
                user_anchor = comment_element.find('span', class_='comment-info').find('a')
                if user_anchor is None:
                    print(f"Skipping comment due to missing user_anchor: {comment_element}")
                    continue

                user_url = user_anchor['href']
                user_info = get_user_info(user_url, headers)
                user = user_anchor.get_text(strip=True)

                comment = comment_element.find('span', class_='short')
                if comment is None:
                    print(f"Skipping comment due to missing comment text: {comment_element}")
                    continue
                comment_text = comment.get_text(strip=True)

                rating = comment_element.find('span', class_='user-stars')
                rating_text = rating['title'] if rating else None

                if user_info:
                    comments.append({
                        'user': user,
                        'comment': comment_text,
                        'rating': rating_text,
                        'user_info': user_info
                    })
            except Exception as e:
                print(f"Error occurred while processing comment: {e}")
                continue

        page += 1
    return comments

def get_full_summary(book_url, headers):
    headers['User-Agent'] = random.choice(USER_AGENTS)
    try:
        response = requests.get(book_url, headers=headers, cookies=COOKIES, proxies={"http": random.choice(PROXIES)})
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))  # 添加延迟
    except requests.RequestException as e:
        print(f"Request to {book_url} failed: {e}")
        return "N/A"

    soup = BeautifulSoup(response.content, 'html.parser')
    full_summary_element = soup.find('span', class_='all hidden')
    if full_summary_element:
        return full_summary_element.get_text(strip=True)
    summary_element = soup.find('div', class_='intro')
    return summary_element.get_text(strip=True) if summary_element else 'N/A'

def get_book_details(book_url, headers, category, subcategory):
    try:
        headers['User-Agent'] = random.choice(USER_AGENTS)
        response = requests.get(book_url, headers=headers, cookies=COOKIES, proxies={"http": random.choice(PROXIES)})
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))  # 添加延迟
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('span', property='v:itemreviewed').get_text(strip=True)
        info_section = soup.find('div', id='info')

        author = None
        publisher = None
        pub_date = None
        pages = None
        price = None
        isbn = None

        # 提取详细信息
        for info in info_section.find_all('span', class_='pl'):
            text = info.get_text(strip=True)
            if '作者' in text:
                author = info.next_sibling.next_sibling.get_text(strip=True)
            elif '出版社' in text:
                publisher_element = info.find_next_sibling('a')
                if publisher_element:
                    publisher = publisher_element.get_text(strip=True)
                else:
                    publisher = info.next_sibling.strip()
            elif '出版年' in text:
                pub_date = info.next_sibling.strip()
            elif '页数' in text:
                pages = info.next_sibling.strip()
            elif '定价' in text:
                price = info.next_sibling.strip()
            elif 'ISBN' in text:
                isbn = info.next_sibling.strip()

        book_id = book_url.split('/')[-2]
        comments = get_all_comments(book_id, headers, max_comments=20)
        summary = get_full_summary(book_url, headers)

        book_details = {
            'Title': title,
            'Author': author,
            'Publisher': publisher,
            'Publication Date': pub_date,
            'Pages': pages,
            'Price': price,
            'ISBN': isbn,
            'Summary': summary,
            'Category': category,
            'Subcategory': subcategory,
            'Comments': comments
        }
        return book_details
    except Exception as e:
        print(f"Error occurred while fetching details for {book_url}: {e}")
        return None

def fetch_books_from_tag(tag_url, headers, category, subcategory, num_books=100, start_index=0):
    books = []
    page = start_index // 20
    fetched_count = start_index
    while fetched_count < num_books:
        url = f"{tag_url}?start={page * 20}"
        print(f"Fetching page {page + 1}: {url}")
        headers['User-Agent'] = random.choice(USER_AGENTS)
        try:
            response = requests.get(url, headers=headers, cookies=COOKIES, proxies={"http": random.choice(PROXIES)})
            response.raise_for_status()
            time.sleep(random.uniform(1, 3))  # 添加延迟
        except requests.RequestException as e:
            print(f"Request to {url} failed: {e}")
            break

        if response.status_code == 418:  # 418状态码表示豆瓣识别为机器人，需要验证码
            print("Encountered captcha, skipping this page.")
            break

        soup = BeautifulSoup(response.content, 'html.parser')

        items = soup.find_all('li', class_='subject-item')
        if not items:
            break

        for item in items:
            if fetched_count >= num_books:
                break
            try:
                detail_url = item.find('a', class_='nbg')['href']
                print(f"Fetching details for {detail_url}")
                book_details = get_book_details(detail_url, headers, category, subcategory)
                if book_details:
                    books.append(book_details)
                    fetched_count += 1
            except Exception as e:
                print(f"Error occurred while processing item: {e}")

        page += 1

    return books

def load_existing_books(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding='utf-8-sig').to_dict('records')
    return []

def save_books_to_csv(books, file_path):
    df = pd.DataFrame(books)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

def load_progress(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_progress(progress, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

def main():
    url = "https://book.douban.com/tag/?view=type&icn=index-sorttags-all"
    headers = {
        'User-Agent': random.choice(USER_AGENTS)
    }
    progress_file = 'progress.json'

    tags = get_all_tags(url, headers)
    if not tags:
        print("No tags found.")
        return

    print(f"Found {len(tags)} tags")

    all_books = load_existing_books('douban_books_detailed_with_comments.csv')
    progress = load_progress(progress_file)

    for category_name, tag_name, tag_url in tags:
        if tag_name in progress and progress[tag_name] == 'completed':
            print(f"Skipping already fetched tag: {tag_name} in category: {category_name}")
            continue

        start_index = progress.get(tag_name, 0)
        print(f"Fetching books for tag: {tag_name} in category: {category_name} starting from index: {start_index}")
        books = fetch_books_from_tag(tag_url, headers, category_name, tag_name, num_books=100, start_index=start_index)
        all_books.extend(books)
        save_books_to_csv(all_books, 'douban_books_detailed_with_comments.csv')
        print(f"Collected {len(books)} books for tag: {tag_name}")

        if len(books) >= 100:
            progress[tag_name] = 'completed'
        else:
            progress[tag_name] = len(books)

        save_progress(progress, progress_file)

    print(f"Total books collected: {len(all_books)}")

    if all_books:
        save_books_to_csv(all_books, 'douban_books_detailed_with_comments.csv')
        print("数据已保存到douban_books_detailed_with_comments.csv文件中")
    else:
        print("No data to save")

if __name__ == "__main__":
    main()