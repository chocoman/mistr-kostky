#!/usr/bin/python

import cv2, cv, sys, getopt, math, Pycluster
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from math import sqrt,acos,pi,degrees
def a(truple):
	return np.array(truple, dtype = "uint8")

def kernel(r): 
	xx, yy = np.mgrid[:r, :r]
	distance = (xx - r/2) ** 2 + (yy - r/2) ** 2
	circle = np.less(distance, (r/2)**2)+0
	return np.array(circle,dtype="uint8")

def clean(image):
#	delete = cv2.dilate(image,kernel(10)) 
#	delete = cv2.erode(delete,kernel(50))
#	delete = cv2.dilate(delete,kernel(80))
	maintain = cv2.erode(image, kernel(4))
	maintain = cv2.dilate(maintain, kernel(12))
	image = image & maintain #& (~delete)
	return image
def intersection(origin1,dir1,origin2,dir2):
	k1=dir1[1]/(dir1[0]+0.0000001) #y=kx+q
	q1=origin1[1]-k1*origin1[0]
	k2=dir2[1]/(dir2[0]+0.0000001)
	q2=origin2[1]-k2*origin2[0]
	if (abs(k1-k2)<0.01):
		print "pruser"
		print dir1
		print origin1
		print dir2
		print origin2
		x = -(q1-q2)/(k1-k2)
		y = k1*x+q1
		print "intersect"
		print [x,y]
	else:
		x = -(q1-q2)/(k1-k2)
	y = k1*x+q1
	return np.array([x,y], dtype=np.int32)
	
def sharpen(polygon): #sharpens corners
	V = [polygon[i][0] for i in range(len(polygon))] #approxPolyDP vraci fuj vec
	E = [V[i]-V[i-1] for i in range(len(V))]
	
	shortest = 0
	while (len(E)>6 and shortest<10):
		shortest = 10000000000
		for i in range(len(V)):
			if (math.hypot(E[i][0],E[i][1])<shortest):
				shortest = math.hypot(E[i][0],E[i][1])
				si = i
		V[si] = intersection(V[si-2],E[si-1],V[(si+1)%len(V)],E[(si+1)%len(E)])
		del V[si-1]
		E = [V[i]-V[i-1] for i in range(len(V))]

	return np.array(V, dtype=np.int32)

def detect(image): #returns sticker information in format [Vertices,epsilon]
	global borders
	ret = []
	image = cv2.erode(image,kernel(2))
	contours, hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for i in range(len(contours)):
		cnt = contours[i]
		
		sticker = h & 0
		cv2.drawContours(sticker, contours, i, 255 ,-1)
		border = cv2.dilate(sticker,kernel(6)) & white
		borderSize = cv2.countNonZero(border)
		borders = borders | border
		center, radius = cv2.minEnclosingCircle(cnt)
		if (radius>2 and borderSize>radius*3):
			hull = cv2.convexHull(cnt)
			hull = sharpen(hull)
			epsilon=0;
			approx=cv2.approxPolyDP(hull,epsilon,1)
			while (len(approx)>4):
				epsilon+=0.1
				approx=cv2.approxPolyDP(hull,epsilon,1)
			if (len(approx)==4 and epsilon<4):
				cv2.drawContours(tetragons, [approx], 0, 255 ,-1)
				cv2.drawContours(selected, contours, i, 255 ,-1)		
				ret.append([approx,epsilon])
			cv2.drawContours(hullimage, [hull], 0, 255 ,-1)
	
	
	return ret

def deviation(a,b):
	sizes = sqrt(a[0]**2+a[1]**2)*sqrt(b[0]**2+b[1]**2)
	if (sizes==0):
		return 0
	dev = acos((a[0]*b[0]+a[1]*b[1])/(sizes+0.0001))

	if (dev<0):
		dev = -dev
	if (dev>pi/2):
		dev = pi-dev
	return degrees(dev)

def distance(sticker1,sticker2): #how different and how far are the tetragons
	V1 = [sticker1[0][k][0] for k in range(len(sticker1[0]))] #approxPolyDP vraci fuj vec
	V2 = [sticker2[0][k][0] for k in range(len(sticker2[0]))] #approxPolyDP vraci fuj vec
	M1 = (V1[0]+V1[1]+V1[2]+V1[3])/4
	M2 = (V2[0]+V2[1]+V2[2]+V2[3])/4
	distance = sum((M1-M2)**2)
	vector1a = ((V1[1]-V1[0])+(V1[2]-V1[3]))/2
	vector2a = ((V2[1]-V2[0])+(V2[2]-V2[3]))/2
	vector1b = ((V1[1]-V1[2])+(V1[0]-V1[3]))/2
	vector2b = ((V2[1]-V2[2])+(V2[0]-V2[3]))/2
	if (vector1a[0]<0):
		vector1a = -vector1a
	if (vector1b[0]<0):
		vector1b = -vector1b
	if (vector2a[0]<0):
		vector2a = -vector2a
	if (vector2b[0]<0):
		vector2b = -vector2b
	diff1 = (vector1a-vector2a)**2+(vector1b-vector2b)**2
	diff2 = (vector1b-vector2a)**2+(vector1a-vector2b)**2
	diff = min(sum(diff1),sum(diff2))
	return sqrt(1*diff + 0*distance)
	
def weight(sticker):
	V = [sticker[0][k][0] for k in range(len(sticker[0]))]
	epsilon=sticker[1]
	dev1 = deviation(V[1]-V[0],V[2]-V[3]) #how much is the tetrahedron similar to rovnobeznik
	dev2 = deviation(V[2]-V[1],V[3]-V[0])
#	weight = 1/((epsilon**2)*(inrarity**2)*(dev1**2)*(dev2**2))
	weight = 10000/((dev1**3)+(dev2**3)+epsilon**2)
	return weight


imgBGR = cv2.imread(sys.argv[1],cv2.CV_LOAD_IMAGE_COLOR)
height, width, _ = imgBGR.shape
if (height>width):
	imgBGR = cv2.resize(imgBGR, (200,200*height/width))
else:
	imgBGR = cv2.resize(imgBGR, (200*width/height,200))
	
imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(imgHSV)
stickers = []
tetragons = h & 0  #hack: blank image
selected = h & 0  #hack: blank image
weighted = h & 0  #hack: blank image
borders = h & 0
hullimage = h & 0
steny = h & 0
print "start"

white = cv2.inRange(imgHSV,a([0,0,10]),a([255,60,255]))

green = cv2.inRange(imgHSV,a([45,100,16]),a([90,255,255]))
green = clean(green)
stickers += detect(green)

blue = cv2.inRange(imgHSV,a([95,150,40]),a([125,255,255]))
blue = clean(blue)
stickers += detect(blue)

orange = cv2.inRange(imgHSV,a([4,120,120]),a([20,255,255]))
orange = clean(orange)
stickers += detect(orange)

yellow = cv2.inRange(imgHSV,a([20,50,100]),a([30,255,255]))
yellow = clean(yellow)
stickers += detect(yellow)

red1 = cv2.inRange(imgHSV,a([0,150,80]),a([3,255,255]))
red2 = cv2.inRange(imgHSV,a([150,150,80]),a([180,255,255]))
red = red1 | red2
red = clean(red)
stickers += detect(red)

W=[weight(st) for st in stickers]
D=[[distance(st1,st2) for st1 in stickers] for st2 in stickers]
tree = Pycluster.treecluster(distancematrix=D)

#FINDING NICEST FACE
cluster_count = 1
while True: #FIXME V tomhle jsem se hrabal a neotestoval
	cluster_count+= 1
	clusters = tree.cut(cluster_count)
	weights=[0 for i in range(cluster_count+1)]
	for i in range(len(stickers)):
		weights[clusters[i]] += W[i]
		#TODO zapocitat i velikost HACK: vzdalenost od nulovyho ctyruhelniku
	maxcluster_weight = 0
	maxcluster_id = 0
	for i in range(cluster_count): #TODO na jeden radek
		if weights[i]>maximum:
			max_id = i
#	maxcluster_maxdistance = ??? #TODO
	if maxcluster_count<=9: # and maxcluster_maxdistance<=100: #TODO vhodnou konstantu
		break


for i in range(len(stickers)):
	if (weight(stickers[i])>5) and clusters[i] == maxcluster_id:
		cv2.drawContours(steny, [stickers[i][0]], 0, 255,-1)

for st in stickers:
	cv2.drawContours(weighted, [st[0]], 0, 10*weight(st),-1)

print "done"

pictures = ['h', 's','v', 'green', 'blue', 'orange', 'yellow', 'white', 'red', 'borders', 'tetragons', 'selected', 'weighted', 'steny']

details = True
if (details):
	for i in xrange(14):
		plt.subplot(3,5,i+1),plt.imshow(eval(pictures[i]),'gray')
		plt.axis("off")
		plt.title(pictures[i])
	imgRGB=mpimg.imread(sys.argv[1])
	plt.subplot(3,5,15),plt.imshow(imgRGB);
else:
	plt.subplot(1,2,1),plt.imshow(steny,'gray')
	plt.axis("off")
	imgRGB=mpimg.imread(sys.argv[1])
	plt.subplot(1,2,2),plt.imshow(imgRGB);


plt.axis("off")
plt.show()
