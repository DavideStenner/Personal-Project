from selenium import webdriver 
from time import sleep 
from selenium.webdriver.chrome.options import Options
import time
from tqdm.notebook import tqdm
from PIL import Image
import os
import glob
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

url_immobiliare = """https://www.immobiliare.it/vendita-case/milano/?criterio=dataModifica&ordine=desc&prezzoMassimo=300000&superficieMinima=80&localiMinimo=3&noAste=1&idMZona[]=10067&idQuartiere[]=11714&idQuartiere[]=10323"""

url_idealista = """https://www.idealista.it/aree/vendita-case/con-prezzo_300000,dimensione_80,trilocali-3,quadrilocali-4,5-locali-o-piu/?ordine=pubblicazione-desc&shape=%28%28sfmtGicwv%40%7DLct%40nuC%7BaC%60NbPu%5BtcAweAzw%40es%40hh%40%29%29"""

gitignore = ['.gitignore']

def delete_old_files(folder):

    files = glob.glob(folder)
    if files:
        for f in files:
            if f not in gitignore:
                os.remove(f)

def resize_top_image(browser, path):
    total_xpath = "/html/body/div[1]/section[1]"
    bottom_xpath = "/html/body/div[1]/section[1]/div[4]/div[1]"
    left_xpath = "/html/body/div[1]/aside/nd-contact-form/section[1]"
    
    anchor_location = browser.find_elements_by_tag_name("nd-read-all")[0].location
    browser.set_window_size(1780, anchor_location['y'])
    
    browser.get_screenshot_as_file(path)
    
    top_location = browser.find_element("xpath", total_xpath).location
    bottom_location = browser.find_element("xpath", bottom_xpath).location
    left_location = browser.find_element("xpath", left_xpath).location
    
    x = top_location['x']
    y = top_location['y']
    w = left_location['x']
    h = bottom_location['y']
    
    image = Image.open(path) 
    image = image.crop((x, y, w, h)) 
    image.save(path) 

def save_top_image(url, folder, name_file = 'banner', time_sleep = 1):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--start-maximized')

    browser = webdriver.Chrome("chromedriver/chromedriver.exe", options = chrome_options)
    browser.get(url)
    home_page = browser.current_window_handle

    number_home = len(browser.find_elements_by_xpath("//p[@class='titolo text-primary']"))
    box_location = "/html/body/div[1]/section[1]"
    image_xpath = "/html/body/div[1]/section[1]/nd-gallery/nd-showcase/figure/nd-showcase/div[1]/div[1]/img"
    
    mapping_url = {}
    
    for i, home in enumerate(range(number_home)):
        save_path = os.path.join(folder, f"{name_file}_{i}.png")

        try:
            banner_intercept = browser.find_elements_by_css_selector("button.close")[0]
            banner_intercept.click()
        except:
            pass

        browser.find_elements_by_xpath("//p[@class='titolo text-primary']")[i].click()

        single_home_window = browser.window_handles[-1]
        browser.switch_to.window(single_home_window)   
        mapping_url[save_path] = browser.current_url

        try:
            cookies_button = browser.find_element("xpath", '/html/body/div[1]/div/button')
            cookies_button.click()
        except:
            pass

        try:
            WebDriverWait(browser, 30).until(
                EC.presence_of_element_located((By.XPATH, image_xpath))
            )
        except:
            pass

        sleep(time_sleep) 

        resize_top_image(browser, save_path)

        browser.close()
        browser.switch_to.window(home_page)

        
    browser.quit()
    
    return mapping_url