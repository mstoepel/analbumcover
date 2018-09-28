# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:16:29 2016
@author: michael
"""

from bs4 import BeautifulSoup
import urllib
import os

#url = 'http://newalbumreleases.net/category/metal/'
#url = 'http://newalbumreleases.net/category/metal/page/10'
#url = 'http://newalbumreleases.net/category/rock/'

url_list = ['http://newalbumreleases.net/category/metal/']
for n in range(4):
    url_list.append('http://newalbumreleases.net/category/metal/page/'+str(n+1))

for n in range(4):
    soup = BeautifulSoup(urllib.request.urlopen(url_list[n]))
    img_links = soup.findAll("div", {"class":"entry"})

# THIS DOWNLOADS THE IMAGES, AND NAMES THEM BASED ON THEIR URL. WANT TO NAME THEM BASED ON ALBUM_GENRE

    st = 'Style: '
    art= 'Artist: '
    alb = 'Album: '
    st2 = '<br/>'

    for img_link in img_links:
        img_url = img_link.img['src']
        str1 = str(img_link.findAll('p')[1])
        str2 = str1[str1.find(art)+11:str1.find(st2,str1.find(art))]
        str3 = str1[str1.find(alb)+10:str1.find(st2,str1.find(alb))]
        str4 = str1[str1.find(st)+7:str1.find(st2,str1.find(st))]
        # file_name = "c:/ML/analbumcover/test/" + str2.replace(" ","") + "_" + str3.replace(" ","") + "_" + str4.replace(" ","") +".jpg"
        file_base = 'c:/ML/analbumcover/album_covers/{}'.format(str4.replace(' ',''))
        print(file_base)
        file_name = "c:/ML/analbumcover/album_covers/{2}/{0}_{1}.jpg".format(str2.replace(' ',''), str3.replace(' ',''), str4.replace(' ', ''))
        if not os.path.exists(file_base):
            print('making it')
            os.makedirs(file_base)
        print(file_name)
        urllib.request.urlretrieve(img_url, file_name)



#for img_link in img_links:
#    print img_link.img['src']