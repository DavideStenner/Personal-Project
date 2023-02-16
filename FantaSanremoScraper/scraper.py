#%%
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import os
from collections import Counter
from time import sleep
import pyautogui
from tqdm import tqdm
import numpy as np
import argparse
from glob import glob

class ScraperFanta():
    def __init__(
        self, number_page_scrape: int=None, pct_scrape: float =.6, 
        backup: int = 500, keep_active_pc_iteration: int = 25,
        check_unique_name: bool=False, 
        selected_league: str = "Campionato Mondiale",
        path_config: str='config.json', path_credential: str ='credential.json',
        test: bool =False, overwrite: bool = True, failsafe: bool = False

    ):
        pyautogui.FAILSAFE = False

        with open(path_credential) as cred_file:
            credential = json.load(cred_file)
        
        with open(path_config) as config_file:
            config = json.load(config_file)
        
        assert selected_league in config["league_dict"].keys()

        self.email = credential['email']
        self.password = credential['password']

        self.backup = backup
        self.test = test
        self.link = config["link"]
        self.league_id = config["league_dict"][selected_league]
        self.wait_time=config["wait_time"]
        self.xpath_dict=config["xpath_dict"]
        self.class_dict=config["class_dict"]
        self.check_unique_name=check_unique_name
        self.num_selection = config["num_selection"]
        
        self.initialize_driver()
        
        self.results = Counter()
        self.captain = Counter()
        self.score_selection = Counter()

        self.unique_team_set = set(['pybranchia'])

        self.number_page_scrape = number_page_scrape
        self.keep_active_pc_iteration = keep_active_pc_iteration
        self.number_element_by_page=config["number_element_by_page"]
        self.pct_scrape=pct_scrape

        self.path_save = f"data/{selected_league}/"

        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save)

        if not os.path.exists(os.path.join(self.path_save, 'backup')):
            os.makedirs(os.path.join(self.path_save, 'backup'))

        if overwrite:
            list_file = glob(
                os.path.join(self.path_save, '**/results*.json'), 
                recursive=True
            )
            for file_path in list_file:
                os.remove(file_path)

    def initialize_driver(self):
        chrome_options = Options()
        if not self.test:
            chrome_options.add_argument('--headless')

        driver = webdriver.Chrome(
            ChromeDriverManager().install(),
            options = chrome_options
        )
        driver.get(self.link)
        self.driver = driver

    def wait_and_click_by(
        self, by_, pattern, 
    ):
        element_ = self.wait_(by_, pattern, EC.element_to_be_clickable, return_element=True)

        actions = webdriver.ActionChains(self.driver)
        actions.move_to_element(element_)
        actions.click(element_)
        actions.perform()

    def wait_(self, by_, pattern, conditions, return_element=False):
        element_ = WebDriverWait(self.driver, self.wait_time).until(
            conditions(
                (by_, pattern)
            )
        )
        if return_element:
            return element_

    def wait_and_find(
        self, by_, pattern
    ):    
        self.wait_(by_, pattern, EC.presence_of_element_located)
        
        return self.driver.find_element(by_, pattern)
    
    def quit(self):
        self.driver.quit()

    
    def login(self):
        self.wait_and_click_by(By.XPATH, self.xpath_dict['coockie'])
        actions = webdriver.ActionChains(self.driver)

        input_email = self.wait_and_find(By.ID, ":r0:")
        actions.move_to_element(input_email)
        actions.perform()
        input_email.send_keys(self.email)

        input_password = self.wait_and_find(By.ID, ':r1:')
        actions.move_to_element(input_password)
        actions.perform()
        input_password.send_keys(self.password)

        self.wait_and_click_by(By.XPATH, self.xpath_dict['accept_credential'])
        self.wait_and_click_by(By.ID, self.league_id)

        if self.number_page_scrape is None:

            #wait element
            self.wait_(By.XPATH, self.xpath_dict['total_team'], EC.presence_of_element_located)
            soup = self.get_html_source()

            total_number_team = int(
                soup.find(
                    'h6', 
                    {"class": self.class_dict['number_total_team']}
                ).getText().replace('Squadre', '')
            )
            used_number_team = int(total_number_team * self.pct_scrape)

            self.number_page_scrape = int(used_number_team//self.number_element_by_page)
            print(f'Number of different teams: {total_number_team}')
            print(f'Number of scraped teams for {self.pct_scrape*100:.1f}%: {used_number_team}')
        
        print(f'Number of pages to scrape: {self.number_page_scrape}')

    def get_html_source(self):
        html_current_page = self.driver.page_source
        soup = BeautifulSoup(html_current_page, features="html.parser")
        return soup

    def get_statistics(self):

        sleep(.5)
        soup = self.get_html_source()

        team_box_list = soup.find_all('div', {"class": self.class_dict['box_info']})

        for team_box in team_box_list:
            team_name = team_box.find('div', {'class': self.class_dict['team_info']}).getText().strip()
            
            if self.check_unique_name:
                if (team_name not in self.unique_team_set) & (team_name != ''):
                    self.unique_team_set.add(team_name)

                    artists_list = [
                        x.get('src')
                        for x in team_box.find_all('img')
                        if 'artists/' in x.get('src')
                    ]

                    artists_counter = Counter(artists_list)

                    for weight, artists in enumerate(artists_list):
                        self.score_selection[artists] += (self.num_selection - weight)

                    self.captain.update(set([artists_list[0]]))
                    self.results.update(artists_counter)
            else:
                artists_list = [
                    x.get('src')
                    for x in team_box.find_all('img')
                    if 'artists/' in x.get('src')
                ]

                artists_counter = Counter(artists_list)

                for weight, artists in enumerate(artists_list):
                    self.score_selection[artists] += (self.num_selection - weight)

                self.captain.update(set([artists_list[0]]))
                self.results.update(artists_counter)

    def get_next_page(self):
        #not necessary to scan every button -> next page is the last loaded
        self.wait_and_click_by(By.XPATH, self.xpath_dict['next_page'])

    def random_sleep(self):
        sleep_time = np.random.uniform(.35, .5, 1)[0]
        sleep(sleep_time)

    def save_results(self, iteration=None):

        save_results = {
            'frequency': self.results,
            'captain': self.captain,
            'weight': self.score_selection,
        }
        path_save = (
            os.path.join(self.path_save, 'backup', f"results_{iteration}.json") 
            if iteration is not None 
            else os.path.join(self.path_save, "results.json")
        )
        with open(
            path_save, 
            "w"
        ) as outfile:
            json.dump(dict(save_results), outfile)

    def keep_pc_active(self):

        pyautogui.press('volumedown')
        sleep(.01)
        pyautogui.press('volumeup')

    def activate_bot(self):
        self.login()

        self.get_statistics()
        self.random_sleep()

        for iteration in tqdm(range(self.number_page_scrape)):
            if (iteration % self.backup == 0) & (iteration > 0):
                self.save_results(iteration)
            
            if (iteration % self.keep_active_pc_iteration == 0) & (iteration > 0):
                self.keep_pc_active()

            self.get_next_page()
            self.get_statistics()
            self.random_sleep()

        self.save_results()
        
# %%
# scraper = ScraperFanta(number_page_scrape = 5, test=True, backup=1)
# scraper.activate_bot()


#%%
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--league', type=str, default="Campionato Mondiale")
    parser.add_argument('--number_page_scrape', type=int, default=None)
    parser.add_argument('--pct_scrape', type=float, default=0.3)
    parser.add_argument('--backup', type=int, default=1000)
    parser.add_argument('--keep_active_pc_iteration', type=int, default=25)
    parser.add_argument('--check_unique_name', action='store_true')
    parser.add_argument('--scrape_all_league', action='store_true')
    args = parser.parse_args()

    if args.scrape_all_league:
        with open('config.json') as config_file:
            league_dict = json.load(config_file)['league_dict']
        
        for league_name, league_id in league_dict.items():
            print(f'\n\nStarting {league_name}')
            
            scraper = ScraperFanta(
                selected_league=league_name,
                check_unique_name=args.check_unique_name,
                number_page_scrape=args.number_page_scrape, pct_scrape=args.pct_scrape, 
                backup = args.backup, keep_active_pc_iteration = args.keep_active_pc_iteration
            )
            scraper.activate_bot()
            scraper.quit()
    else:
        print(f'\n\nStarting {args.league}')
        scraper = ScraperFanta(
            selected_league=args.league,
            check_unique_name=args.check_unique_name,
            number_page_scrape=args.number_page_scrape, pct_scrape=args.pct_scrape, 
            backup = args.backup, keep_active_pc_iteration = args.keep_active_pc_iteration
        )
        scraper.activate_bot()
        scraper.quit()

