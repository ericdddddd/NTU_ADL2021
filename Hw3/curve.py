import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10]
y_1 = [0.1319,0.1593,0.1738,0.1770,0.1799,0.1973,0.2063,0.2111,0.2145,0.2174]
y_2 = [0.042,0.05367,0.061,0.0621,0.0637,0.0713,0.0756,0.0789,0.0804,0.0813]
y_3 = [0.1298,0.1543,0.171,0.174,0.1763,0.1909,0.2005,0.2041,0.2061,0.2094]

Data_1, = plt.plot(x,y_1,'r-.^',label='rogue-1') #畫線
Data_2, = plt.plot(x, y_2, 'g--*',label='rogue-2') #畫線
Data_3, = plt.plot(x,y_3,'b-.^',label='rogue-L') #畫線

plt.tick_params(axis='both', color='green')
plt.legend(handles=[Data_1, Data_2, Data_3])
plt.show() #顯示繪製的圖形
