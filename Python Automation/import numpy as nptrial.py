# importing module
from pandas import *
import csv
import numpy as np
# reading CSV file
dataInput = read_csv("D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\rectfh_bot_node_with_coord.csv")
dataOutput = "D:\materials of JI\\4903robotic arm\\ansys working\\ansys scripting\\output_rectfh_bottomsurf_nodes.csv"
# converting column data to list
NodeNum = dataInput['Node Number'].tolist()
LOC_X = dataInput['X Location (mm)'].tolist()
LOC_Y= dataInput['Y Location (mm)'].tolist()
LOC_Z = dataInput['Z Location (mm)'].tolist()
# width a, height b
input_displacement=[]
a = 17.5
b = 12.5
z_0 = -19.537
y_MAX = 12
z_MAX = 12
radius = 2
angle_Max=15
Nb = 10
Na = 10
bb = b / Nb
aa = a / Na
Nsel = []
def FindID_Byloc(x,y,z,r,N):
    '''Given a location find an id'''
    N.clear()
    #FindID_Byloc(25-Fypos[i]+7.8473, Fxpos[i] + 9.1152, z_0, radius,Nsel)
    string = ':' + str(x) + ', ' + str(y) +', ' +  str(z) + ', ' + str(radius)
    # transfer to global coords:
    temp = z
    z = x - 10.7
    x = y - 90.1
    y = temp
    N.append(string)
    if y == -1:
        N.append('null')
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
        for n in Ng:
            rowid = []
            rowid.append(n)
            spamwriter.writerow(rowid)
def judgement(x,y,dx,dy,da):
    x_cen_trans = x  *np.cos(da) - y  *np.sin(da) + dx
    y_cen_trans = x  *np.sin(da) + y  *np.cos(da) + dy
    if (x_cen_trans  > -b and x_cen_trans  < b) and (y_cen_trans >-a and y_cen_trans < a):
        return 0
    return 1
def convert(delta, y_bar, x_bar):
    '''convert position delta to force acting point'''
    for dx in np.arange(-z_MAX, z_MAX+1, 1):
        for dy in np.arange(-y_MAX, y_MAX+1, 1):
            for da in np.arange(-angle_Max,angle_Max+3,3):
                delta.append([dx, dy, da])
                #ccw as positive for da
                square = []
                A = 0
                A_times_x = 0
                A_times_y = 0
                for i in np.arange(0, 2 * b , 2*bb):
                    for j in np.arange(0, 2* a , 2*aa):
                        square.append([i, j])
                        if judgement(i + bb - b, j + aa - a, dx,dy,da/180*np.pi):
                                # All four points of the square are outside the hole
                                A +=  4 * aa * bb
                                A_times_x += 4 * aa * bb * (i + bb)
                                A_times_y += 4 * aa * bb * (j + aa)
                if A!=0:
                    x_cm = A_times_x / A
                    y_cm = A_times_y / A
                    x_bar.append(x_cm)
                    y_bar.append(y_cm)
                else:
                    x_bar.append(-1)
                    y_bar.append(-1)
with open(dataOutput, 'w') as f:
    f.truncate()
Nsel_list = []
Fxpos =[]
Fypos = []
delta = []
convert(delta, Fypos, Fxpos)
for i in range(0, len(delta)):
        FindID_Byloc(Fxpos[i], Fypos[i], z_0, radius,Nsel)
        #print(Nsel)
        Nsel_list.append(Nsel)
        Write_data(dataOutput, Nsel)
print(len(Nsel_list))


