from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import os
import urllib.request
import argparse
import sys
from multiprocessing import Process, Value


class GoogleImageScrapper:

    def __init__(self,timeout=2):
        # maximum time spent downloading the image (in second)
        self.timeout = timeout

    def __call__(self,searchterm,n_max=100,verbose=0):
        self.searchterm = searchterm
        self.root_save = "results"
        self.save_path = self.root_save + "/" + self.searchterm
        self.n_max = n_max
        self._init()
        self._scrapping(verbose)

    def _init(self):
        self.url = "https://www.google.co.in/search?q="+self.searchterm+"&source=lnms&tbm=isch"

        # NEED TO DOWNLOAD CHROMEDRIVER, insert path to chromedriver inside parentheses in following line
        chrome_options = Options()
        # to avoid the browser to open
        chrome_options.add_argument("--headless")
        self.browser = webdriver.Chrome("drivers/chromedriver-mac", chrome_options=chrome_options)
        self.browser.get(self.url)
        self.header = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        self.counter = 0
        self.succounter = Value('i',0)

        if not os.path.exists(self.root_save):
            os.mkdir(self.root_save)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # to see more images
        self._scroll_down()

    def _get_succounter(self):
        return self.succounter.value

    def _scroll_down(self):
        """
        Scroll down the webpage to see more images
        """
        for _ in range(5 * self.n_max):
            self.browser.execute_script("window.scrollBy(0,10000)")

    def _download_and_save_image_function(self,url,imgtype,verbose):
        """
        Download and save the image located at the url
        :param url: url where to find the image
        :param imgtype: type of the image
        :param verbose: verbose level (see _scrapping for more detailled description)
        """
        try:
            raw_image = urllib.request.urlopen(url).read()
            File = open(os.path.join(self.save_path, self.searchterm + "_" + str(self.counter) + "." + imgtype), "wb")
            File.write(raw_image)
            File.close()
            # should be done with a lock but only one active thread here
            self.succounter.value += 1
        except Exception as e:
            if verbose == 2:
                print(e)

    def _cleanup(self):
        """
        Clean up the result folder (removes files with no extensions)
        :return:
        """
        for file in os.listdir(self.save_path):
            # if no extension, deletes the the file
            if file[-1] == ".":
                os.remove(os.path.join(self.save_path,file))

    def _scrapping(self,verbose):
        """
        Download all the images from the search
        :param verbose: 0, 1 or 2
                0 for no verbose at all
                1 for inline verbose
                2 to keep all the details of the execution
        """
        for x in self.browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
            if self.succounter.value >= self.n_max:
                break
            url = json.loads(x.get_attribute('innerHTML'))["ou"]
            if verbose == 2:
                print("Total Count:", self.counter)
                print("Succsessful Count:", self.succounter.value)
                print("URL:", url)
            elif verbose == 1:
                print("\rTotal Count: %d Successful Count: %d" % (self.counter, self.succounter.value),end="")
            imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]

            # We create a Process that will download the image
            action_process = Process(target=self._download_and_save_image_function, args=(url,imgtype,verbose))

            # We start the process and we block for timeout seconds.
            action_process.start()
            action_process.join(timeout=self.timeout)

            self.counter = self.counter + 1

        if verbose >= 1:
            print()
            print(self.succounter.value, "pictures succesfully downloaded")
        self.browser.close()

        self._cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # search_term
    parser.add_argument('-s','--searchterm',type=str,required=True,
                        help="term to search inside google images")
    #  n_max
    parser.add_argument('-n','--nmax', default=10, type=int,
                        help='number of images that will ideally be downloaded (keep it under 1000)')
    # timeout
    parser.add_argument('-t','--timeout', default=2, type=int,
                        help='number of seconds for the download to be completed')
    # verbose
    parser.add_argument('-v','--verboselevel', default=1, type=int,
                        help='level of verbose (0, 1 or 2)')

    args = parser.parse_args()
    return args

def main(args):
    g = GoogleImageScrapper(timeout=args.timeout)
    g(args.searchterm, n_max=args.nmax, verbose=args.verboselevel)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        print('Please provide some parameters...')
        sys.exit()

    main(args)











