# importing module
from pandas import *
import csv
import numpy as np
# reading CSV file
dataInput = read_csv("D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\filenode cylin.csv")
dataOutput = "D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\output 1125 data.csv"
# converting column data to list
NodeNum = dataInput['Node Number'].tolist()
LOC_X = dataInput['X Location (mm)'].tolist()
LOC_Y= dataInput['Y Location (mm)'].tolist()
LOC_Z = dataInput['Z Location (mm)'].tolist()
x_0 = 19.491
y_MAX = 30
z_MAX = 10
radius = 0.5
Nsel = []
def FindID_Byloc(x,y,z,r,N):
    '''Given a location find an id'''
    N.clear()
    string = ':' + str(x) + ', ' + str(y) +', ' +  str(z) + ', ' + str(radius)
    N.append(string)
    if y == -1:
        N.append('null')
        return
    for node in NodeNum:
        index = NodeNum.index(node)
        xloc = LOC_X[index]
        yloc = LOC_Y[index]
        zloc = LOC_Z[index]
        Distance = ((xloc - x) ** 2 + (yloc - y) ** 2 + (zloc - z) ** 2) ** 0.5
        if Distance < r:
            N.append(str(node))
            break
def Write_data(filename, Ng):
    with open(filename, 'a') as file:
        spamwriter = csv.writer(file, lineterminator='\n')
        char = []
        #char.append(str(x) + ',' + str(y) +',' +  str(radius))
        #spamwriter.writerow(char)
        for n in Ng:
            rowid = []
            rowid.append(n)
            spamwriter.writerow(rowid)
def convert(delta, y_bar, x_bar):
    '''convert position delta to force acting point'''
    a = 30
    b = 10
    for dx in np.arange(-z_MAX,z_MAX + 1,1):
        for dy in np.arange(-y_MAX,y_MAX + 1,1):
            if (dx == 0 or dy == 0):
                delta.append([dx, dy])
                y_abs = abs(dy)
                x_abs = abs(dx)
                # y/x is the Fypos/Fxpos in original coordinate
                y = (y_abs*b*(a - y_abs/2)+(a - y_abs) ** 2 * x_abs / 2)/(y_abs * b + x_abs * (a - y_abs))
                #print (y)
                x = ((y_abs*b ** 2) / 2+(b - x_abs / 2) * x_abs * (a - y_abs))/(y_abs * b + x_abs * (a - y_abs))
                #print(x)
                # convert coordinate system when dx or dy < 0
                if dx >= 0 and dy >= 0 and dx + dy != 0:
                    y_bar.append(y)
                    x_bar.append(x)
                if dx < 0 and dy >= 0:  
                    y_bar.append(y)
                    x_bar.append(b - x)      
                if dx < 0 and dy < 0:  
                    y_bar.append(a - y)
                    x_bar.append(b - x)  
                if dx >= 0 and dy < 0:  
                    y_bar.append(a - y)
                    x_bar.append(x)  
                if dx == 0 and dy == 0:
                    y_bar.append(-1)
                    x_bar.append(-1)
# clear the file
with open(dataOutput, 'w') as f:
    f.truncate()
Nsel_list = []
Fxpos =[]
Fypos = []
delta = []
convert(delta, Fypos, Fxpos)
print('Fxpos: ')
#for i in Fxpos:
#    print(i)
print(len(Fxpos))
print('Fypos: ')
for i in Fypos:
    print(i)
print(len(Fypos))
print('delta: ')
for i in delta:
    print (i)
print(len(delta))
for i in range(0, len(Fypos)):
        FindID_Byloc(x_0, Fypos[i], Fxpos[i],radius,Nsel)
        #print(Nsel)
        Nsel_list.append(Nsel)
        Write_data(dataOutput, Nsel)
print(len(Nsel_list))
# printing list data
'''
print('Node number:', NodeNum[0])
print('x location:', LOC_X[0])
print('y location:', LOC_Y[0])
print('z location:', LOC_Z[0])
print(NodeNum.index(17965))'''
#import csv
#filename = "D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\filenode.csv"
#with open(filename, 'r') as file:
 #   reader = csv.reader(file, delimiter=' ', quotechar='|')
  #  for row in reader:
   #     print(row[0])
#def LocgetID(x, y, radius):
