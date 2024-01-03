import numpy as np
import os
import subprocess
# get test scene
with open('demo_test.txt','r') as f:
    filelist = f.readlines()
filelist = [name.strip() for name in filelist]

demo_path = './demo.py'
for filename in filelist:
    try:
        subprocess.run(['python',demo_path,"--filename",filename],check=True)
    except subprocess.CalledProcessError as e:
        print("Error {}".format(e))
    print(filename)
    # break