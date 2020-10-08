import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

def get_pages(url, args):
    
    response = requests.get(url, headers = {'user-agent': args.header}, timeout = args.timeout)

    soup = BeautifulSoup(response.text, 'lxml')
    return(soup)


def selenium_builder(args):
    chrome_options = webdriver.ChromeOptions()

    chrome_options.add_argument('headless')

    browser = webdriver.Chrome(executable_path = args.chrome_path, options = chrome_options)
    browser.delete_all_cookies()
    browser.implicitly_wait(args.time_implicit_w8)
    
    return(browser)

def selenium_proxy_connector(url, args, script):

    browser = selenium_builder(args)
    
    try:
        browser.get(url)
        result = browser.execute_script("return aItems;")
        browser.quit()
        
    except:
        browser.quit()
        time.sleep(args.time_w8)
    
        result = selenium_proxy_connector(url, args, script)
        
    return(result)
