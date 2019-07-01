import cv2
import numpy as np
import pandas as pd

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img=np.array(Image.open('D:\\picture processing\\pic1\\5.jpg').convert('L'))
img2=np.array(Image.open('D:\\picture processing\\pic1\\5.jpg').convert('L'))
plt.figure("lena")
arr=img.flatten()
arr2=img2.flatten()
n, bins, patches = plt.hist(arr, bins=256,density=1, facecolor='green', alpha=0.75)
n2, bins, patches = plt.hist(arr2, bins=256,density=1, facecolor='green', alpha=0.75)
m=n.reshape(1,256)
m2=n2.reshape(1,256)
# plt.show()
# print(n)
p=np.vstack((m,m2))
# np.concatenate((a,b),axis=0)

data_df = pd.DataFrame(p)
writer = pd.ExcelWriter('Save_Excel.xlsx')
data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
writer.save()

np.savetxt('new.csv', m, delimiter = ',')

cv2.waitKey(0)
cv2.destroyAllWindows()
