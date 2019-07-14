# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
from imageio import imread, imsave

book='3157556/'

tree = ET.parse(book+'docElements.xml')
root = tree.getroot()
idx = 0
pad = 4
labels=[]
idf=0
for image in root.iter('image'):
    pg=image.attrib.get('src')
    idf=int(image.attrib.get('id'))
    print(idf)
    if (idf>=0):
        print('in')
        img = imread(book+str(pg)+'.png')
        for zone in image.iter('zone'):
            c=0
            for point in zone.iter('point'):
                if c==0:
                    y1= point.attrib.get('y')
                    x1= point.attrib.get('x')
                if c==2:
                    y2= point.attrib.get('y')
                    x2= point.attrib.get('x')
                c=c+1
            id=int(zone.attrib.get('id'))
            label=None
            for segment in root.iter('segment'):
                tid=int(segment.attrib.get('id'))
                if tid==id:
                    label= segment[1].text
            if label is None:
                label='mislabel'
                print ('there is a mismatch label')
            
            label=label.strip()                
            labels.append(label)
            if not os.path.exists('segment'+book+label):
                os.makedirs('segment'+book+label)
            
            roi = img[int(y1)-pad:int(y2)+pad,int(x1)-pad:int(x2)+pad]
            direc='segment'+book+label
            imsave(direc+'/' + str(idf) +'-'+str(idx)+ '.png', roi)
            idx = idx + 1
    else:
        continue


    
