# import requests
# from lxml.html import fromstring
# import urllib3
# import numpy as np
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry
# import time 
# from bs4 import BeautifulSoup
# from selenium import webdriver

# def get_proxies(ip_len = None, url = 'https://free-proxy-list.net/'):
#     urllib3.disable_warnings()    
#     response = requests.get(url)
#     parser = fromstring(response.text)
    
#     ip_parser = parser.xpath('//tbody/tr')
        
#     proxies = list()
#     for i in ip_parser:
#         if (len(i.xpath('.//td[7][contains(text(),"yes")]'))>0) & (len(i.xpath('.//td[5][contains(text(),"elite proxy")]'))>0):
#         #Grabbing IP and corresponding PORT
#             proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
#             proxies += [proxy]
    
#     if (ip_len is not None) & (len(proxies)>ip_len):
#         proxies = proxies[:ip_len]
#     np.random.shuffle(proxies) 
    
#     return proxies

# def session_builder(proxy, args):
#     session = requests.session()
#     session.proxies = {}
#     session.proxies = {"http": 'http://' + proxy, "https": 'https://' + proxy}

#     session.cookies.clear()
    
#     #evita proxy difettosi non si connettano.
#     retry = Retry(connect = args.connection, backoff_factor = args.backof_f)
#     adapter = HTTPAdapter(max_retries = retry)
#     session.mount('http://', adapter)
#     session.mount('https://', adapter)

#     return(session)

# def proxy_connector(proxies, url, args, headers):
    
#     while len(proxies) == 0:
#         proxies = get_proxies(args.MAX_LEN)
#         print('\nProxy list ended...')

#     session = session_builder(proxies[0], args)
    
#     try:
#         response = session.get(url, headers = headers, timeout = args.MAX_W8, verify = False)
        
#     except:
#         waiter = args.pause_factor[0] * np.random.random(1) + args.pause_factor[1]
#         time.sleep(waiter)
    
#         del proxies[0]
#         response, proxies = proxy_connector(proxies = proxies, url = url, args = args, headers = headers)
        
#     return(response, proxies)


# def get_pages(proxies, url, args):
    
#     #define header
#     headers = {}
#     headers['user-agent'] = args.header

#     response, proxies = proxy_connector(proxies = proxies, url = url, args = args, headers = headers)
#     if response.status_code == 400:
#         del proxies[0]
#         response, proxies = proxy_connector(proxies = proxies, url = url, args = args, headers = headers)
        
#     soup = BeautifulSoup(response.text, 'html5lib')
#     return(soup, proxies)

# def selenium_builder(proxy, executable_path, time_w8 = 60):
#     chrome_options = webdriver.ChromeOptions()

#     chrome_options.add_argument('headless')
#     chrome_options.add_argument('--proxy-server=%s' %proxy)

#     browser = webdriver.Chrome(executable_path = executable_path, options = chrome_options)
#     browser.delete_all_cookies()
#     browser.implicitly_wait(time_w8)
    
#     return(browser)

# def selenium_proxy_connector(proxies, url, args, script = 'return aItems;'):
#     while len(proxies) == 0:
#         proxies = get_proxies(args.MAX_LEN)
#         print('\nProxy list ended...')

#     browser = selenium_builder(proxy = proxies[0], executable_path = args.chrome_path)
    
#     try:
#         browser.get(url)
#         result = browser.execute_script("return aItems;")
#         browser.quit()
        
#     except:
#         browser.quit()
#         waiter = args.PAUSE_RANGE[0] * np.random.random(1) + args.PAUSE_RANGE[1]
#         time.sleep(waiter)
    
#         del proxies[0]
#         result = selenium_proxy_connector(proxies = proxies, url = url, args = args)
        
#     return(result)
