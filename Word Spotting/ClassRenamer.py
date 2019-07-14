# -*- coding: utf-8 -*-

import os
book='3157556segr'
os.chdir(book)
label=0
for foldername in sorted(os.listdir(str(os.getcwd()))):
    l=len(str(label))
    z=5-l
    a=''
    for i in range(z):
        a+='0'
    os.rename(foldername,a+str(label))
    label=label+1
    
