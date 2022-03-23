import numpy as np
import matplotlib.pyplot as plt

def generator_colors(basicColor,deltaColor):
    (r,g,b)=basicColor
    (dr,dg,db)=deltaColor
    return color_normalization(basicColor,[(r-1*dr,g-1*dg,b-1*db),(r-2*dr,g-2*dg,b-2*db),(r-3*dr,g-3*dg,b-3*db)],(r+2*dr,g+2*dg,b+2*db))
def color_normalization(basicColor0,optionalColors0,threadColor0):
    return [tuple([i/255 for i in basicColor0]),
            [(i[0]/255,i[1]/255,i[2]/255) for i in optionalColors0],
            tuple([i/255 for i in threadColor0])]
def rand_color(optionalColors):
    return optionalColors[np.random.randint(0,len(optionalColors))]
def random_range(a,b):
    return a+(b-a)*np.random.rand()
def random_position(dataRange):
    (lower0,upper0)=dataRange
    upper=upper0*(0.6+0.3*np.random.rand())
    lower=upper*(0.5+0.2*np.random.rand())
    return (lower,upper)
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
def pattern_stripe_nonuniform_half(direction,width,flatSize,n,color1):
    points0=[0,width*flatSize]+[width*flatSize+(i/(2*n))*width*(1-flatSize) for i in range(1,2*n+1)]
    if direction=='left': points=points0
    if direction=='right': points=np.sort([size-i for i in points0])
    intervals0=[(int(points[i]),int(points[i+1])) for i in range(len(points)-1)]
    intervals=[intervals0[i] for i in range(len(intervals0)) if not i%2]
    colors=[color1 for i in intervals]
    return list(zip(intervals,colors))
def pattern_smooth(dataRange,flatSize,color1,color2):
    lenght=dataRange[-1]-dataRange[0]
    points=list(range(int(dataRange[0]),int(dataRange[0]+lenght*(1-flatSize)/2)))+\
           [int(dataRange[0]+lenght*(1-flatSize)/2),int(dataRange[-1]-lenght*(1-flatSize)/2)]+\
           list(range(int(dataRange[-1]-lenght*(1-flatSize)/2+1),int(dataRange[-1]+1)))
    intervals=get_intervals(points)
    colors=get_colors('gradient',len(intervals),color1,color2)
    return list(zip(intervals,colors))
def add_multiline(linewidth,n=2):
    flag=True
    pos1=[]
    while flag:
        flag=False
        pos0=np.sort(np.random.choice(range(int(0.25*size),int(0.75*size)),n,False))
        for i in range(n-1):
            if (pos0[i+1]-pos0[i])/size<0.2: flag=True
    for i in pos0:
        r=np.random.randint(1,3)
        pos1.append((i-size*(linewidth/2)*(2*r-1),i+size*(linewidth/2)*(2*r-1)))
    return pos1


(basicColor,optionalColors,threadColor)=generator_colors((90,90,120),(25,25,25))
# basicColor0=(235,211,213)
# threadColor0=(255,255,255)
# optionalColors0=[(174,155,149),(234,225,216),(178,166,160)]
# (basicColor,optionalColors,threadColor)=color_normalization(basicColor0,optionalColors0,threadColor0)

size=600
gap=4
colorModes=['alternant','gradient']
xy_same=False


layer_X=np.zeros((2*size,2*size))
layer_Y=np.zeros((2*size,2*size))
for i in range(2*size):
    for j in range(2*size):
        if (i+j)%(2*gap)>=gap: layer_X[i,j]=1
        else: layer_Y[i,j]=1
pattern_X=np.ones((size,size,3))
pattern_Y=np.ones((size,size,3))
pattern_all=np.zeros((2*size,2*size,3))
for rgb in range(3):
    pattern_X[:,:,rgb]=np.ones((size,size))*basicColor[rgb]
    pattern_Y[:,:,rgb]=np.ones((size,size))*basicColor[rgb]
style_X=[]
style_Y=[]
multiline_X=add_multiline(1/25,2)
multiline_Y=add_multiline(1/25,2)
thread_X=add_multiline(1/100,2)
thread_Y=add_multiline(1/100,2)

''' ---------- custom ---------- '''
drange=(0,size)
drange=(0,size)
for i in range(3):
    drange=random_position(drange)
    a=np.random.randint(4)
    if a==0 or a==1: style_X+=pattern_smooth(drange,random_range(0,0.6),basicColor,rand_color(optionalColors))
    if a==2: style_X+=pattern_stripe_uniform(drange,np.random.randint(2,5),rand_color(optionalColors),rand_color(optionalColors))
    if a==3: style_X+=pattern_stripe_nonuniform(drange,random_range(0,0.6),random_range(0.2,1),np.random.randint(2,5),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
style_X+=pattern_stripe_nonuniform_half('left',size*random_range(0.05,0.1),0.75,np.random.randint(3),basicColor)
style_X+=pattern_stripe_nonuniform_half('right',size*random_range(0.04,0.08),0.2,np.random.randint(2),rand_color(optionalColors))
for multiline in multiline_X:
    if np.random.randint(2): style_X+=pattern_stripe_uniform(multiline,int(((multiline[1]-multiline[0])/(size/25)+1)/2),rand_color(optionalColors))
for thread in thread_X:
    if np.random.randint(2): style_X+=pattern_stripe_uniform(thread,int(((thread[1]-thread[0])/(size/100)+1)/2),threadColor)

drange=(0,size)
for i in range(5):
    drange=random_position(drange)
    a=np.random.randint(4)
    if a==0 or a==1: style_Y+=pattern_smooth(drange,random_range(0,0.6),basicColor,rand_color(optionalColors))
    if a==2: style_Y+=pattern_stripe_uniform(drange,np.random.randint(2,5),rand_color(optionalColors),rand_color(optionalColors))
    if a==3: style_Y+=pattern_stripe_nonuniform(drange,random_range(0,0.6),random_range(0.2,1),np.random.randint(2,5),colorModes[np.random.randint(2)],rand_color(optionalColors),rand_color(optionalColors))
style_Y+=pattern_stripe_nonuniform_half('left',size*random_range(0.05,0.1),0.75,np.random.randint(3),basicColor)
style_Y+=pattern_stripe_nonuniform_half('right',size*random_range(0.04,0.08),0.2,np.random.randint(2),rand_color(optionalColors))
for multiline in multiline_Y:
    if np.random.randint(2): style_Y+=pattern_stripe_uniform(multiline,int(((multiline[1]-multiline[0])/(size/25)+1)/2),rand_color(optionalColors))
for thread in thread_Y:
    if np.random.randint(2): style_Y+=pattern_stripe_uniform(thread,int(((thread[1]-thread[0])/(size/100)+1)/2),threadColor)
''' ---------- custom ---------- '''

#  1 4
#  2 3
if xy_same:
    for i in style_Y:
        (pos,color)=i
        for rgb in range(3):
            pattern_Y[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
    piece1=pattern_Y.copy()
    for rgb in range(3): pattern_Y[:,:,rgb]=np.flipud(pattern_Y[:,:,rgb])
    piece2=pattern_Y.copy()
    for rgb in range(3): pattern_Y[:,:,rgb]=np.fliplr(pattern_Y[:,:,rgb])
    piece3=pattern_Y.copy()
    for rgb in range(3): pattern_Y[:,:,rgb]=np.flipud(pattern_Y[:,:,rgb])
    piece4=pattern_Y.copy()
    pattern_Y_all=np.zeros_like(pattern_all)
    for rgb in range(3):
        pattern_Y_all[:,:,rgb]=np.concatenate((np.concatenate((piece1[:,:,rgb],piece2[:,:,rgb])),np.concatenate((piece4[:,:,rgb],piece3[:,:,rgb]))),1)
    pattern_X_all=pattern_Y_all.copy()
else:
    for i in style_X:
        (pos,color)=i
        for rgb in range(3):
            pattern_X[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
    piece1=pattern_X.copy()
    for rgb in range(3): pattern_X[:,:,rgb]=np.flipud(pattern_X[:,:,rgb])
    piece2=pattern_X.copy()
    for rgb in range(3): pattern_X[:,:,rgb]=np.fliplr(pattern_X[:,:,rgb])
    piece3=pattern_X.copy()
    for rgb in range(3): pattern_X[:,:,rgb]=np.flipud(pattern_X[:,:,rgb])
    piece4=pattern_X.copy()
    pattern_X_all=np.zeros_like(pattern_all)
    for rgb in range(3):
        pattern_X_all[:,:,rgb]=np.concatenate((np.concatenate((piece1[:,:,rgb],piece2[:,:,rgb])),np.concatenate((piece4[:,:,rgb],piece3[:,:,rgb]))),1)

    for i in style_Y:
        (pos,color)=i
        for rgb in range(3):
            pattern_Y[pos[0]:pos[1],:,rgb]=np.ones((size,size))[pos[0]:pos[1],:]*color[rgb]
    piece1=pattern_Y.copy()
    for rgb in range(3): pattern_Y[:,:,rgb]=np.flipud(pattern_Y[:,:,rgb])
    piece2=pattern_Y.copy()
    for rgb in range(3): pattern_Y[:,:,rgb]=np.fliplr(pattern_Y[:,:,rgb])
    piece3=pattern_Y.copy()
    for rgb in range(3): pattern_Y[:,:,rgb]=np.flipud(pattern_Y[:,:,rgb])
    piece4=pattern_Y.copy()
    pattern_Y_all=np.zeros_like(pattern_all)
    for rgb in range(3):
        pattern_Y_all[:,:,rgb]=np.concatenate((np.concatenate((piece1[:,:,rgb],piece2[:,:,rgb])),np.concatenate((piece4[:,:,rgb],piece3[:,:,rgb]))),1)
for rgb in range(3):
    pattern_X_all[:,:,rgb]=pattern_X_all[:,:,rgb]*layer_X
    pattern_Y_all[:,:,rgb]=np.rot90(pattern_Y_all[:,:,rgb],3)*layer_Y
    pattern_all[:,:,rgb]=pattern_X_all[:,:,rgb]+pattern_Y_all[:,:,rgb]
img=np.tile(pattern_all,(2,3,1))
plt.imshow(img)
plt.show()