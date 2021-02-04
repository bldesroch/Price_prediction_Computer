import selenium 
from selenium import webdriver
from selenium.webdriver import Firefox
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep, time
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
from selenium.common.exceptions import NoSuchElementException
import json
import csv
import os

class Scrolling:
    def __init__(self, url):
        self.driver = webdriver.Firefox()
        self.driver.get(url)
        sleep(3)
        self.search =self.driver.find_element_by_xpath('//*[@id="footer_tc_privacy_button"]').click()
        self.search = self.driver.find_element_by_xpath('/html/body/div[1]/header/div/div/div[1]/div/div[3]/div[1]/input')
        self.search.send_keys("ordinateur portable")
        sleep(2)
        self.search.send_keys(Keys.RETURN)
        
    def page(self, path):
        debut = time()
        with open(path, 'a') as f:
            for a in self.driver.find_elements_by_class_name('prdtBlocInline.jsPrdtBlocInline'):
                ann = Annonce(a)
                f.write(ann.to_json())
                f.write("\n")
        return time() - debut
    
        
    def page_suivante(self):
        sleep(11)
        self.driver.find_element_by_class_name('btBlue.jsNxtPage').click()

        
class Annonce:


    def __init__(self, annonce):
        self.set_prix(annonce)
        self.set_lien(annonce)
        self.set_nom(annonce)
        self.set_desc(annonce)

    def __str__(self):
        return f"""
lien        : {self.lien}
prix        : {self.prix}
nom         : {self.nom}
description : {self.desc}
"""
        
    def set_nom(self, annonce):
        """Donne le nom de l'ordinateur"""
        n= annonce.find_element_by_class_name('prdtBILTit')
        self.nom = n.text    


    def set_prix(self, annonce):
        """Donne le prix de l'ordinateur"""
        p = annonce.find_element_by_class_name("price")
        self.prix = p.text


    def set_desc(self, annonce):
        """Donne la description de l'annonce de l'ordinateur"""
        d = annonce.find_element_by_class_name('prdtBILDesc.jsPrdtBILLink')
        self.desc = d.text.splitlines()


    def set_lien(self, annonce):
        """Donne le lien de l'annonce"""
        h = annonce.find_element_by_class_name('prdtBILDetails')
        l= h.find_element_by_tag_name('a')
        self.lien = l.get_attribute("href")
    
    def to_json(self):
        return json.dumps(self.__dict__)

def scraping():
    url = "https://www.cdiscount.com/"
    fichier = Path(".").resolve() / "brute1.json"
    nav = Scrolling(url)
    while True:
        duree = nav.page(fichier)
        print("Les données page {} ont été recupérées".format(nav.driver.current_url))
        if duree < 5:
            sleep(6 - int(duree))
        try:
            nav.page_suivante()
        except NoSuchElementException:
            break
        sleep(5)
    nav.driver.quit()