import numpy as np
import matplotlib.pyplot as plt


def rand_color(optionalColors):
    return optionalColors[np.random.randint(0,len(optionalColors))]
def color_normalization(basicColor0,optionalColors0,threadColor0):
    return [tuple([i/255 for i in basicColor0]),
            [(i[0]/255,i[1]/255,i[2]/255) for i in optionalColors0],
            tuple([i/255 for i in threadColor0])]
def generator_colors(basicColor,deltaColor):
    (r,g,b)=basicColor
    (dr,dg,db)=deltaColor
    return color_normalization(basicColor,[(r-1*dr,g-1*dg,b-1*db),(r-2*dr,g-2*dg,b-2*db),(r-3*dr,g-3*dg,b-3*db)],(r+2*dr,g+2*dg,b+2*db))
def get_intervals(points):
    points.sort()
    intervals=[(int(points[i]),int(points[i+1])) for i in range(len(points)-1)]
    return intervals
def get_colors(colorMode,n,color1,color2):
    if colorMode=='gradient':
        x=(n-1)/2
        return [(color1[0]*abs(i-x)/(x)+color2[0]*(1-abs(i-x)/x),
                 color1[1]*abs(i-x)/(x)+color2[1]*(1-abs(i-x)/x),
                 color1[2]*abs(i-x)/(x)+color2[2]*(1-abs(i-x)/x)) for i in range(n)]
    if colorMode=='alternant':
        return [color1 if not i%2 else color2 for i in range(n)]
def pattern_stripe_uniform(dataRange,n,color1,color2):
    points=np.linspace(dataRange[0],dataRange[-1],2*n,endpoint=True)
    intervals=get_intervals(points)
    colors=get_colors('alternant',len(intervals),color1,color2)
    return list(zip(intervals,colors))
def pattern_stripe_nonuniform(dataRange,flatSize,offset,n,colorMode,color1,color2):
    lenght=dataRange[-1]-dataRange[0]
    points=list(np.linspace(dataRange[0],dataRange[0]+lenght*(1-flatSize)*(1/(1+offset)),n+1,endpoint=True))+\
           list(np.linspace(dataRange[-1]-lenght*(1-flatSize)*(offset/(1+offset)),dataRange[-1],n+1,endpoint=True))
    intervals=get_intervals(points)
    colors=get_colors(colorMode,len(intervals),color1,color2)
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
        if not np.random.randint(2): pos0.append(i[np.random.randint(2)])
    for i in pos0:
        r=np.random.randint(1,3)
        pos3.append((i-size/200*(2*r-1),i+size/200*(2*r-1)))
    return pos3


# (basicColor,optionalColors,threadColor)=generator_colors((100,90,120),(30,30,25))
basicColor0=(250,220,160)
threadColor0=(255,255,255)
optionalColors0=[(220,180,120),(190,150,110),(160,120,100)]
(basicColor,optionalColors,threadColor)=color_normalization(basicColor0,optionalColors0,threadColor0)

size=400
gap=4
colorModes=['alternant','gradient']
xy_same=False


layer_X=np.zeros((size,size))
layer_Y=np.zeros((size,size))
for i in range(size):
    for j in range(size):
        if (i+j)%(2*gap)>gap: layer_X[i,j]=1
        else: layer_Y[i,j]=1
pattern_X=np.ones((size,size,3))
pattern_Y=np.ones((size,size,3))
pattern=np.zeros((size,size,3))
for rgb in range(3):
    pattern_X[:,:,rgb]=np.ones((size,size))*basicColor[rgb]
    pattern_Y[:,:,rgb]=np.ones((size,size))*basicColor[rgb]
style_X=[]
style_Y=[]
background_X=add_wide(2)
background_Y=add_wide(2)
wide_X=add_wide(3)
wide_Y=add_wide(3)
narrow_X=add_narrow(wide_X)
narrow_Y=add_narrow(wide_Y)
thread_X=add_thread(wide_X)
thread_Y=add_thread(wide_Y)

''' ---------- custom ---------- '''
if np.random.randint(2): style_X+=pattern_smooth(background_X[0],0,basicColor,rand_color(optionalColors))
if np.random.randint(2): style_X+=pattern_smooth(background_X[1],0,basicColor,rand_color(optionalColors))
if np.random.randint(2): style_X+=pattern_smooth(wide_X[0],np.random.randint(0,40)/100,basicColor,rand_color(optionalColors))
if np.random.randint(2): style_X+=pattern_stripe_nonuniform(wide_X[1],np.random.randint(40,80)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],basicColor,rand_color(optionalColors))
if np.random.randint(2): style_X+=pattern_stripe_nonuniform(wide_X[2],np.random.randint(40,80)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],basicColor,rand_color(optionalColors))
if np.random.randint(2): style_X+=pattern_smooth(narrow_X.pop(np.random.randint(len(narrow_X))),np.random.randint(0,50)/100,basicColor,rand_color(optionalColors))
if np.random.randint(2): style_X+=pattern_stripe_nonuniform(narrow_X.pop(np.random.randint(len(narrow_X))),np.random.randint(0,40)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
if np.random.randint(2): style_X+=pattern_stripe_nonuniform(narrow_X.pop(np.random.randint(len(narrow_X))),np.random.randint(0,40)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
for thread in thread_X:
    if np.random.randint(2): style_X+=pattern_stripe_uniform(thread,int(((thread[1]-thread[0])/(size/100)+1)/2),threadColor,basicColor)

if np.random.randint(3): style_Y+=pattern_smooth(background_Y[0],0,basicColor,rand_color(optionalColors))
if np.random.randint(3): style_Y+=pattern_smooth(background_Y[1],0,basicColor,rand_color(optionalColors))
if np.random.randint(3): style_Y+=pattern_smooth(wide_Y[0],np.random.randint(0,40)/100,basicColor,rand_color(optionalColors))
if np.random.randint(3): style_Y+=pattern_stripe_nonuniform(wide_Y[1],np.random.randint(40,80)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],basicColor,rand_color(optionalColors))
if np.random.randint(3): style_Y+=pattern_stripe_nonuniform(wide_Y[2],np.random.randint(40,80)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],basicColor,rand_color(optionalColors))
if np.random.randint(3): style_Y+=pattern_smooth(narrow_Y.pop(np.random.randint(len(narrow_Y))),np.random.randint(0,50)/100,basicColor,rand_color(optionalColors))
if np.random.randint(3): style_Y+=pattern_stripe_nonuniform(narrow_Y.pop(np.random.randint(len(narrow_Y))),np.random.randint(0,40)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
if np.random.randint(3): style_Y+=pattern_stripe_nonuniform(narrow_Y.pop(np.random.randint(len(narrow_Y))),np.random.randint(0,40)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
for thread in thread_Y:
    if np.random.randint(3): style_Y+=pattern_stripe_uniform(thread,int(((thread[1]-thread[0])/(size/100)+1)/2),threadColor,basicColor)
if not np.random.randint(6): style_Y+=pattern_stripe_uniform([(0,size/2),(size/2,size)][np.random.randint(2)],4,rand_color(optionalColors),basicColor)
''' ---------- custom ---------- '''

if xy_same:
    for i in style_X:
        (pos,color)=i
        for rgb in range(3):
            pattern_X[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
    pattern_Y=pattern_X.copy()
else:
    for i in style_X:
        (pos,color)=i
        for rgb in range(3):
            pattern_X[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
    for i in style_Y:
        (pos,color)=i
        for rgb in range(3):
            pattern_Y[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
for rgb in range(3):
    pattern_X[:,:,rgb]=pattern_X[:,:,rgb]*layer_X
    pattern_Y[:,:,rgb]=np.rot90(pattern_Y[:,:,rgb],3)*layer_Y
    pattern[:,:,rgb]=pattern_X[:,:,rgb]+pattern_Y[:,:,rgb]
img=np.tile(pattern,(3,4,1))
plt.imshow(img)
plt.show()