# run multiple lle.py cases in parallel

import os
import time

for i in range(4):
    os.system("python lle.py &")
    time.sleep(20)

