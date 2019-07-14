# -*- coding: utf-8 -*-
import os
import shutil
book='segment3157556'
folder='segment315755610'
for foldername in os.listdir(folder):
    if not foldername.startswith('.'):
        if len(os.listdir(folder+'/'+foldername))>=20:
            shutil.move(folder+'/'+foldername,book+'20/'+foldername )
