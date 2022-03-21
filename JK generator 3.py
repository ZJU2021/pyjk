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
def pattern_stripe_uniform(dataRange,n,color1,color2=None):
    points=np.linspace(dataRange[0],dataRange[-1],2*n,endpoint=True)
    intervals=get_intervals(points)
    if color2:
        colors=get_colors('alternant',len(intervals),color1,color2)
        intervals_colors=list(zip(intervals,colors))
        return intervals_colors
    else:
        colors=get_colors('alternant',len(intervals),color1,color1)
        intervals_colors=list(zip(intervals,colors))
        return [intervals_colors[i] for i in range(len(intervals_colors)) if not i%2]
def pattern_stripe_nonuniform(dataRange,flatSize,offset,n,colorMode,color1,color2=None):
    lenght=dataRange[-1]-dataRange[0]
    points=list(np.linspace(dataRange[0],dataRange[0]+lenght*(1-flatSize)*(1/(1+offset)),n+1,endpoint=True))+\
           list(np.linspace(dataRange[-1]-lenght*(1-flatSize)*(offset/(1+offset)),dataRange[-1],n+1,endpoint=True))
    intervals=get_intervals(points)
    if color2:
        colors=get_colors(colorMode,len(intervals),color1,color2)
        intervals_colors=list(zip(intervals,colors))
        return intervals_colors
    else:
        colors=get_colors(colorMode,len(intervals),color1,color1)
        intervals_colors=list(zip(intervals,colors))
        return [intervals_colors[i] for i in range(len(intervals_colors)) if not i%2]
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
        pos0=np.sort(np.random.choice(range(size),2*n,False))
        for i in range(n):
            if not 0.1*unit<(pos0[2*i+1]-pos0[2*i])/size<2*unit: flag=True
        for i in range(1,n):
            if not 0.02*unit<(pos0[2*i]-pos0[2*i-1])/size<0.2*unit: flag=True
        if not 0.3*unit<pos0[0]/size<0.6*unit: flag=True
        if not 0.3*unit<(size-pos0[-1])/size<0.6*unit: flag=True
        pos1=[(pos0[2*i],pos0[2*i+1]) for i in range(n)]
    return pos1
def add_narrow(pos1):
    pos2=[]
    for i in pos1:
        flag=True
        (lower,upper)=i
        while flag:
            flag=False
            pos0=np.sort(np.random.choice(range(lower+1,upper-1),2,False))
            if not 0.1<(pos0[1]-pos0[0])/(upper-lower)<0.9: flag=True
        pos2.append((pos0[0],pos0[1]))
    return pos2
def add_multiline(n=2):
    flag=True
    pos3=[]
    while flag:
        flag=False
        pos0=np.sort(np.random.choice(range(size),n,False))
        for i in range(n-1):
            if (pos0[i+1]-pos0[i])/size<0.2: flag=True
        if pos0[0]/size<0.1: flag=True
        if (size-pos0[-1])/size<0.1: flag=True
    for i in pos0:
        r=np.random.randint(1,4)
        pos3.append((i-size/50*(2*r-1),i+size/50*(2*r-1)))
    return pos3
def add_thread(pos1):
    pos0=[]
    pos4=[]
    for i in pos1:
        if not np.random.randint(2): pos0.append(i[np.random.randint(2)])
    for i in pos0:
        r=np.random.randint(1,3)
        pos4.append((i-size/200*(2*r-1),i+size/200*(2*r-1)))
    return pos4


(basicColor,optionalColors,threadColor)=generator_colors((80,100,80),(25,30,25))
# basicColor0=(235,211,213)
# threadColor0=(255,255,255)
# optionalColors0=[(174,155,149),(234,225,216),(178,166,160)]
# (basicColor,optionalColors,threadColor)=color_normalization(basicColor0,optionalColors0,threadColor0)

size=400
gap=4
colorModes=['alternant','gradient']
xy_same=True


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
multiline_X=add_multiline(2)
multiline_Y=add_multiline(2)
thread_X=add_thread(wide_X)
thread_Y=add_thread(wide_Y)

''' ---------- custom ---------- '''
for count in range(len(background_X)):
    if not np.random.randint(3+count): style_X+=pattern_smooth(background_X.pop(np.random.randint(len(background_X))),0,basicColor,rand_color(optionalColors))
for count in range(len(wide_X)):
    if not np.random.randint(3): style_X+=pattern_stripe_nonuniform(wide_X.pop(np.random.randint(len(wide_X))),np.random.randint(40,80)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],basicColor,rand_color(optionalColors))
    else: style_X+=pattern_smooth(wide_X.pop(np.random.randint(len(wide_X))),np.random.randint(40,80)/100,basicColor,rand_color(optionalColors))
for count in range(len(narrow_X)):
    if not np.random.randint(3+count): style_X+=pattern_stripe_nonuniform(narrow_X.pop(np.random.randint(len(narrow_X))),np.random.randint(0,40)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
for multiline in multiline_X:
    if np.random.randint(2): style_X+=pattern_stripe_uniform(multiline,int(((multiline[1]-multiline[0])/(size/25)+1)/2),rand_color(optionalColors))
for thread in thread_X:
    if np.random.randint(2): style_X+=pattern_stripe_uniform(thread,int(((thread[1]-thread[0])/(size/100)+1)/2),threadColor)

for count in range(len(background_Y)):
    if not np.random.randint(3+count): style_Y+=pattern_smooth(background_Y.pop(np.random.randint(len(background_Y))),0,basicColor,rand_color(optionalColors))
for count in range(len(wide_Y)):
    if not np.random.randint(3): style_Y+=pattern_stripe_nonuniform(wide_Y.pop(np.random.randint(len(wide_Y))),np.random.randint(40,80)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],basicColor,rand_color(optionalColors))
    else: style_Y+=pattern_smooth(wide_Y.pop(np.random.randint(len(wide_Y))),np.random.randint(40,80)/100,basicColor,rand_color(optionalColors))
for count in range(len(narrow_Y)):
    if not np.random.randint(3+count): style_Y+=pattern_stripe_nonuniform(narrow_Y.pop(np.random.randint(len(narrow_Y))),np.random.randint(0,40)/100,np.random.randint(25,400)/100,np.random.randint(1,4),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
for multiline in multiline_Y:
    if np.random.randint(2): style_Y+=pattern_stripe_uniform(multiline,int(((multiline[1]-multiline[0])/(size/25)+1)/2),rand_color(optionalColors))
for thread in thread_Y:
    if np.random.randint(2): style_Y+=pattern_stripe_uniform(thread,int(((thread[1]-thread[0])/(size/100)+1)/2),threadColor)
''' ---------- custom ---------- '''

if xy_same:
    for i in style_Y:
        (pos,color)=i
        for rgb in range(3):
            pattern_Y[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
    pattern_X=pattern_Y.copy()
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