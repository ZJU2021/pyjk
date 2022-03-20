import numpy as np
import matplotlib.pyplot as plt

''' color scheme (first: basic color, last: thread color) '''
colors0=[(100,100,160),(180,60,120),(180,100,200),(40,20,100),(160,40,100),(0,0,0)]  # colors input
colors=[(color[0]/255,color[1]/255,color[2]/255) for color in colors0]  # colors conversion (do not change)

''' options '''
n=800  # size of tile
gap=8  # size of thread
wide=3  # number of wide stripes (recommend:3)
xy_same=False  # whether the pattern is same along axis of X&Y
percent=1/wide  # a coefficient for later usage (do not change)


''' what you need to change is shown above '''
''' do not modify content below '''


''' add wide stripes '''
def add_wide(n):
    flag=True  # whether to discard the generated pattern
    while flag:
        flag=False
        pos0=np.sort(np.random.choice(range(1,n-1),2*wide,False))  # generate a random position
        for i in range(wide):  # width of wide stripe
            if not (n*percent)*0.3<pos0[2*i+1]-pos0[2*i]<(n*percent)*0.9: flag=True
        for i in range(1,wide):  # width of gap between stripes
            if not (n*percent)*0.15<pos0[2*i]-pos0[2*i-1]<(n*percent)*0.6: flag=True
        if not (n*percent)*0.15<pos0[0]<(n*percent)*0.3: flag=True  # width of top margin
        if not (n*percent)*0.15<n-pos0[-1]<(n*percent)*0.3: flag=True  # width of bottom margin
        pos1=[(pos0[2*i],pos0[2*i+1]) for i in range(wide)]  # a reasonable position
    return pos1

''' add narrow stripes '''
def add_narrow(pos1):
    pos2=[]
    for i in pos1:
        (lower,upper)=i
        flag=True
        while flag:
            flag=False
            pos0=np.sort(np.random.choice(range(lower+1,upper-1),2,False))  # generate a random position
            if not pos0[0]-lower>(upper-lower)*0.2: flag=True  # width of top margin
            if not upper-pos0[1]>(upper-lower)*0.2: flag=True  # width of bottom margin
            if not (upper-lower)*0.05<pos0[1]-pos0[0]<(upper-lower)*0.1: flag=True  # width of narrow stripe
        pos2.append((pos0[0],pos0[1]))  # a reasonable position
    return pos2

''' add 2 layers '''
layer1=np.zeros((n,n))
layer2=np.zeros((n,n))
for i in range(n):  # basic pattern
    for j in range(n):
        if (i+j)%(2*gap)>gap: layer1[i,j]=1
        else: layer2[i,j]=1

''' add 2 patterns '''
ones=np.ones((n,n))
pattern1=np.ones((n,n,3))
pattern2=np.ones((n,n,3))
for rgb in range(3):  # draw basic color
    pattern1[:,:,rgb]=ones*colors[0][rgb]
    pattern2[:,:,rgb]=ones*colors[0][rgb]

''' add 2 position '''
pos1_1=add_wide(n)
pos2_1=add_wide(n)
pos1_2=add_narrow(pos1_1)
pos2_2=add_narrow(pos2_1)

''' draw pattern 1&2 '''
for i in pos1_1:  # draw color of wide stripes in pattern1
    a=np.random.randint(1,len(colors0)-1)
    for rgb in range(3):
        pattern1[i[0]:i[1],:,rgb]=ones[i[0]:i[1],:]*colors[a][rgb]
for i in pos2_1:  # draw color of wide stripes in pattern2
    a=np.random.randint(1,len(colors0)-1)
    for rgb in range(3):
        pattern2[i[0]:i[1],:,rgb]=ones[i[0]:i[1],:]*colors[a][rgb]
for i in pos1_2:  # draw color of narrow stripes in pattern1
    if i[1]-i[0]<0.03*n: a=np.random.randint(-1,1)
    else: a=0
    for rgb in range(3):
        pattern1[i[0]:i[1],:,rgb]=ones[i[0]:i[1],:]*colors[a][rgb]
for i in pos2_2:  # draw color of narrow stripes in pattern2
    if i[1]-i[0]<0.03*n: a=np.random.randint(-1,1)
    else: a=0
    for rgb in range(3):
        pattern2[i[0]:i[1],:,rgb]=ones[i[0]:i[1],:]*colors[a][rgb]

''' combine patterns '''
pattern=np.zeros((n,n,3))  # final pattern
if xy_same: pattern2=pattern1.copy()
for rgb in range(3):
    pattern1[:,:,rgb]=pattern1[:,:,rgb]*layer1
    pattern2[:,:,rgb]=np.rot90(pattern2[:,:,rgb],3)*layer2
    pattern[:,:,rgb]=pattern1[:,:,rgb]+pattern2[:,:,rgb]
img=np.tile(pattern,(3,4,1))  # tile the final pattern

''' show the final pattern '''
plt.imshow(img)
plt.show()
