import numpy as np
import matplotlib.pyplot as plt


def rand_color(optionalColors):
    return optionalColors[np.random.randint(0,len(optionalColors))]
def generator_colors(basicColor,deltaColor):
    (r,g,b)=basicColor
    (x,y,z)=deltaColor
    return [[(r-1*x,g-1*y,b-1*z),(r-2*x,g-2*y,b-2*z),(r-3*x,g-3*y,b-3*z)],(r+2*x,g+2*y,b+2*z)]
def get_intervals(points):
    points.sort()
    intervals=[(int(points[i]),int(points[i+1])) for i in range(len(points)-1)]
    return intervals
def get_colors(ctype,n,color1,color2):
    if ctype=='gradient':
        x=(n-1)/2
        return [(color1[0]*abs(i-x)/(x)+color2[0]*(1-abs(i-x)/x),
                 color1[1]*abs(i-x)/(x)+color2[1]*(1-abs(i-x)/x),
                 color1[2]*abs(i-x)/(x)+color2[2]*(1-abs(i-x)/x)) for i in range(n)]
    if ctype=='alternant':
        return [color1 if not i%2 else color2 for i in range(n)]
def pattern_stripe(dataRange,n,color1,color2):
    points=np.linspace(dataRange[0],dataRange[-1],2*n,endpoint=True)
    intervals=get_intervals(points)
    colors=get_colors('alternant',len(intervals),color1,color2)
    return list(zip(intervals,colors))
def pattern_gradient(dataRange,flatSize,offset,n,color1,color2):
    lenght=dataRange[-1]-dataRange[0]
    points=list(np.linspace(dataRange[0],dataRange[0]+lenght*(1-flatSize)*(1/(1+offset)),n+1,endpoint=True))+\
           list(np.linspace(dataRange[-1]-lenght*(1-flatSize)*(offset/(1+offset)),dataRange[-1],n+1,endpoint=True))
    intervals=get_intervals(points)
    colors=get_colors('gradient',len(intervals),color1,color2)
    return list(zip(intervals,colors))
def pattern_gradient_stripe(dataRange,flatSize,offset,n,color1,color2):
    lenght=dataRange[1]-dataRange[0]
    points=list(np.linspace(dataRange[0],dataRange[0]+lenght*(1-flatSize)*(1/(1+offset)),n+1,endpoint=True))+\
           list(np.linspace(dataRange[-1]-lenght*(1-flatSize)*(offset/(1+offset)),dataRange[-1],n+1,endpoint=True))
    intervals=get_intervals(points)
    colors=get_colors('alternant',len(intervals),color1,color2)
    return list(zip(intervals,colors))
def pattern_smooth(dataRange,flatSize,color1,color2):
    lenght=dataRange[-1]-dataRange[0]
    points=list(range(int(dataRange[0]),int(dataRange[0]+lenght*(1-flatSize)/2)))+\
           [int(dataRange[0]+lenght*(1-flatSize)/2),int(dataRange[-1]-lenght*(1-flatSize)/2)]+\
           list(range(int(dataRange[-1]-lenght*(1-flatSize)/2+1),int(dataRange[-1]+1)))
    intervals=get_intervals(points)
    colors=get_colors('gradient',len(intervals),color1,color2)
    return list(zip(intervals,colors))
def add_wide(n=3):
    flag=True
    unit=1/n
    while flag:
        flag=False
        pos0=np.sort(np.random.choice(range(1,size-1),2*n,False))
        for i in range(n):
            if not 0.2*unit<(pos0[2*i+1]-pos0[2*i])/size<2.5*unit: flag=True
        for i in range(1,n):
            if not 0.05*unit<(pos0[2*i]-pos0[2*i-1])/size<0.3*unit: flag=True
        if not pos0[0]/size<0.3*unit: flag=True
        if not (size-pos0[-1])/size<0.3*unit: flag=True
        pos1=[(pos0[2*i],pos0[2*i+1]) for i in range(n)]
    return pos1
def add_narrow(pos1):
    pos2=[]
    for i in pos1:
        (lower,upper)=i
        flag=True
        while flag:
            flag=False
            pos0=np.sort(np.random.choice(range(lower+1,upper-1),2,False))
            if not 0.1<(pos0[1]-pos0[0])/(upper-lower)<0.6: flag=True
        pos2.append((pos0[0],pos0[1]))
    return pos2
def add_thread(pos1):
    pos0=[]
    pos3=[]
    for i in pos1:
        if not np.random.randint(3): pos0.append(i[np.random.randint(2)])
    for i in pos0:
        r=np.random.randint(1,3)
        pos3.append((i-size/100*r,i+size/100*r))
    return pos3


# (optionalColors0,threadColor0)= generator_colors((150,90,90),(25,25,25))
basicColor0=(250,220,160)
threadColor0=(250,250,250)
optionalColors0=[(220,180,120),(190,150,110),(160,120,100)]
basicColor=tuple([i/255 for i in basicColor0])
threadColor=tuple([i/255 for i in threadColor0])
optionalColors=[(i[0]/255,i[1]/255,i[2]/255) for i in optionalColors0]

size=400
gap=4
xy_same=False


layer_X=np.zeros((size,size))
layer_Y=np.zeros((size,size))
style_X=[]
style_Y=[]
pattern_X=np.ones((size,size,3))
pattern_Y=np.ones((size,size,3))
pattern=np.zeros((size,size,3))

background_X=add_wide(2)[np.random.randint(2)]
background_Y=add_wide(2)[np.random.randint(2)]
wide_X=add_wide(3)
wide_Y=add_wide(3)
narrow_X=add_narrow(wide_X)
narrow_Y=add_narrow(wide_Y)
thread_X=add_thread(wide_X)
thread_Y=add_thread(wide_Y)

style_X+=pattern_smooth(background_X,np.random.randint(0,30)/100,basicColor,rand_color(optionalColors))
style_X+=pattern_smooth(wide_X[0],np.random.randint(0,30)/100,basicColor,rand_color(optionalColors))
style_X+=pattern_gradient(wide_X[1],np.random.randint(30,60)/100,np.random.randint(25,400)/100,np.random.randint(3,9),basicColor,rand_color(optionalColors))
style_X+=pattern_gradient_stripe(wide_X[2],np.random.randint(30,60)/100,np.random.randint(50,200)/100,np.random.randint(2,8),basicColor,rand_color(optionalColors))
style_Y+=pattern_smooth(background_Y,np.random.randint(0,30)/100,basicColor,rand_color(optionalColors))
style_Y+=pattern_smooth(wide_Y[0],np.random.randint(0,30)/100,basicColor,rand_color(optionalColors))
style_Y+=pattern_gradient(wide_Y[1],np.random.randint(30,60)/100,np.random.randint(25,400)/100,np.random.randint(3,9),basicColor,rand_color(optionalColors))
style_Y+=pattern_gradient_stripe(wide_Y[2],np.random.randint(30,60)/100,np.random.randint(50,200)/100,np.random.randint(2,8),basicColor,rand_color(optionalColors))
if not np.random.randint(3): style_X+=pattern_smooth(narrow_X.pop(np.random.randint(len(narrow_X))),np.random.randint(0,50)/100,basicColor,rand_color(optionalColors))
if not np.random.randint(3): style_X+=pattern_gradient(narrow_X.pop(np.random.randint(len(narrow_X))),np.random.randint(0,30)/100,np.random.randint(25,400)/100,np.random.randint(2,6),rand_color(optionalColors),rand_color(optionalColors))
if not np.random.randint(3): style_X+=pattern_gradient_stripe(narrow_X.pop(np.random.randint(len(narrow_X))),np.random.randint(0,30)/100,np.random.randint(50,200)/100,np.random.randint(1,3),rand_color(optionalColors),rand_color(optionalColors))
if not np.random.randint(3): style_Y+=pattern_smooth(narrow_Y.pop(np.random.randint(len(narrow_Y))),np.random.randint(10,50)/100,basicColor,rand_color(optionalColors))
if not np.random.randint(3): style_Y+=pattern_gradient(narrow_Y.pop(np.random.randint(len(narrow_Y))),np.random.randint(0,30)/100,np.random.randint(25,400)/100,np.random.randint(2,6),rand_color(optionalColors),rand_color(optionalColors))
if not np.random.randint(3): style_Y+=pattern_gradient_stripe(narrow_Y.pop(np.random.randint(len(narrow_Y))),np.random.randint(0,30)/100,np.random.randint(50,200)/100,np.random.randint(1,3),rand_color(optionalColors),rand_color(optionalColors))
for thread in thread_X: style_X+=pattern_stripe(thread,int((thread[1]-thread[0])/(size/50)),threadColor,basicColor)
for thread in thread_Y: style_Y+=pattern_stripe(thread,int((thread[1]-thread[0])/(size/50)),threadColor,basicColor)

for i in range(size):
    for j in range(size):
        if (i+j)%(2*gap)>gap: layer_X[i,j]=1
        else: layer_Y[i,j]=1
for rgb in range(3):
    pattern_X[:,:,rgb]=np.ones((size,size))*basicColor[rgb]
    pattern_Y[:,:,rgb]=np.ones((size,size))*basicColor[rgb]
for i in style_X:
    (pos,color)=i
    for rgb in range(3):
        pattern_X[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
for i in style_Y:
    (pos,color)=i
    for rgb in range(3):
        pattern_Y[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]

if xy_same: pattern_Y=pattern_X.copy()
for rgb in range(3):
    pattern_X[:,:,rgb]=pattern_X[:,:,rgb]*layer_X
    pattern_Y[:,:,rgb]=np.rot90(pattern_Y[:,:,rgb],3)*layer_Y
    pattern[:,:,rgb]=pattern_X[:,:,rgb]+pattern_Y[:,:,rgb]
img=np.tile(pattern,(3,4,1))
plt.imshow(img)
plt.show()