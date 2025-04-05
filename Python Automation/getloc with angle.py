import numpy as np
# width a, height b
a = 12.5
b = 12.5
input_displacement=[]
y_MAX = 12
z_MAX = 12
# width bb and height aa of the small rectangles
angle_Max=15
Nb = 20
Na = 20
bb = b / Nb
aa = a / Na
# A-area of overlap; A_times_x-area multiply x_center; ...
Floc = []
def judgement(x,y,dx,dy,da):
    #transformation operation to the point
    x_cen_trans = x  *np.cos(da) - y  *np.sin(da) + dx
    y_cen_trans = x  *np.sin(da) + y  *np.cos(da) + dy
    if (x_cen_trans  > -b and x_cen_trans  < b) and (y_cen_trans >-a and y_cen_trans < a):
        return 0
    return 1
for dx in np.arange(-z_MAX, z_MAX+1, 1):
    for dy in np.arange(-y_MAX, y_MAX+1, 1): 
        for da in np.arange(-angle_Max,angle_Max+3,3):
            input_displacement.append([dx, dy, da])
            #ccw as positive for da
            square = []
            A = 0
            A_times_x = 0
            A_times_y = 0
            for i in np.arange(0, 2 * b , 2*bb):
                for j in np.arange(0, 2* a , 2*aa):
                    square.append([i, j])
                    if judgement(i + bb - b, j + aa - a, dx,dy,da/180*np.pi):
                            # center of the square outside the hole region
                            A +=  4 * aa * bb
                            A_times_x += 4 * aa * bb * (i + bb)
                            A_times_y += 4 * aa * bb * (j + aa)
            if A!=0:
                x_cm = A_times_x / A
                y_cm = A_times_y / A
                Floc.append([x_cm, y_cm])
            else:
                Floc.append([-1, -1])
#            print([x_cm,y_cm])
#print (len(Floc))
print(Floc)
print(len(input_displacement))
