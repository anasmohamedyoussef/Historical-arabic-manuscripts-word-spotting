# -*- coding: utf-8 -*-
import os
import shutil
book='segment3157556'
folder='segment315755620'
for foldername in os.listdir(folder):
    if not foldername.startswith('.'):
        if len(os.listdir(folder+'/'+foldername))>=100:
            shutil.move(folder+'/'+foldername,book+'100/'+foldername )
