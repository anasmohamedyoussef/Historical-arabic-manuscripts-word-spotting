# -*- coding: utf-8 -*-
import os
import shutil
book='segment3157556'
folder='segment3157556'
for foldername in os.listdir(folder):
    if not foldername.startswith('.'):
        if len(os.listdir(folder+'/'+foldername))>=10:
            shutil.move(folder+'/'+foldername,book+'10/'+foldername )
