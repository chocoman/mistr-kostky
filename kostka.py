#!/usr/bin/python

import cv2, cv, sys, getopt, math, Pycluster
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from math import sqrt,acos,pi,degrees,sin,radians
from time import clock

class stickerinfo:
	def __init__(self, Vertices, epsilon):
		self.V = None 
		self.E = None
		vectora = None
		vectorb = None
		self.center = None
		self.epsilon = None
		self.size = None
		self.weight = None

		self.V = [Vertices[k][0] for k in range(len(Vertices))]
		V = self.V #alias
		self.E = [self.V[i]-self.V[i-1] for i in range(len(self.V))]
		self.epsilon = epsilon
		self.center = (V[0]+V[1]+V[2]+V[3])/4
		vectora = ((V[1]-V[0])+(V[2]-V[3]))/2
		vectorb = ((V[1]-V[2])+(V[0]-V[3]))/2
		if (vectora[0]<0):
			vectora = -vectora
		if (vectorb[0]<0):
			vectorb = -vectorb
		self.size = abs(vectora[0]*vectorb[1]-vectora[1]*vectorb[0])
		if (deviation(vectora,vectorb)<0):
			vectora, vectorb = vectorb, vectora
		self.vectora = vectora
		self.vectorb = vectorb
		dev1 = deviation(V[1]-V[0],V[2]-V[3]) #how much is the tetrahedron similar to rovnobeznik
		dev2 = deviation(V[2]-V[1],V[3]-V[0])
		self.weight = 255/(1+abs(dev1)**2+abs(dev2)**2+epsilon**2)

def a(truple):
	return np.array(truple, dtype = "uint8")

def kernel(r): 
	xx, yy = np.mgrid[:r, :r]
	distance = (xx - r/2) ** 2 + (yy - r/2) ** 2
	circle = np.less(distance, (r/2)**2)+0
	return np.array(circle,dtype="uint8")

def clean(image):
	maintain = cv2.erode(image, kernel(4))
	maintain = cv2.dilate(maintain, kernel(12))
	image = image & maintain
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
	while (len(E)>7 and shortest<6):
		shortest = 10000000000
		for i in range(len(V)):
			if (math.hypot(E[i][0],E[i][1])<shortest):
				shortest = math.hypot(E[i][0],E[i][1])
				si = i
		V[si] = intersection(V[si-2],E[si-1],V[(si+1)%len(V)],E[(si+1)%len(E)])
		del V[si-1]
		E = [V[i]-V[i-1] for i in range(len(V))]

	return np.array(V, dtype=np.int32)

def detect(image): #returns list of detected stickers
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
		if (radius>2 and borderSize>radius*2):
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
				st = stickerinfo(approx, epsilon)
				ret.append(st)
			cv2.drawContours(hullimage, [hull], 0, 255 ,-1)
	return ret

def deviation(a,b):
	sizes = sqrt(a[0]**2+a[1]**2)*sqrt(b[0]**2+b[1]**2)
	if (sizes==0):
		return 0
	dev = acos((a[0]*b[0]+a[1]*b[1])/(sizes+0.0001))
	dev = abs(dev)
	if (dev>pi/2):
		dev = pi-dev
	return degrees(dev)

def distance(sticker1,sticker2): #how different and how far are the tetragons
	V1 = sticker1.V
	V2 = sticker2.V
	distance = sum((sticker1.center-sticker2.center)**2)
	diff = sum((sticker1.vectora-sticker2.vectora)**2+(sticker1.vectorb-sticker2.vectorb)**2)
	return sqrt(1*diff + 0*distance)

def vectorsize(vector):
	return sqrt(vector[0]**2+vector[1]**2)

def maxdistance(cluster_id):
	maximum=0;
	for i in range(len(stickers)):
		if clusters[i] != cluster_id:
			continue
		for j in range(len(stickers)):
			if clusters[j] != cluster_id:
				continue
			if D[i][j] >= maximum:
				maximum = D[i][j]
	return maximum
def find_face(cluster_id):
	face_stickers = []
	for i in range(len(stickers)):
		if clusters[i]==cluster_id and stickers[i].weight>1:
			face_stickers.append(stickers[i])
	a = sum([st.vectora for st in face_stickers])/float(len(face_stickers)) #TODO zatim pouzivam trivialni projekci
	a *= 1.3
	b = sum([st.vectorb for st in face_stickers])/float(len(face_stickers))
	b *= 1.3
	mindeviation = 100000000
	for A in face_stickers: #find best fit to grid with fixed origin A
		deviation = 0
		for B in face_stickers:
			#linear equation B-A = p*a+q*b
			S = B.center-A.center
			q = (S[1]-S[0]*a[1]/a[0])/(-b[0]*a[1]/a[0]+b[1])
			p = S[0]/a[0]-q*b[0]/a[0]
			p,q = round(p),round(q)
			deviation += vectorsize(S-(p*a+p*b))**2 #distance on display
		if deviation < mindeviation:
			mindeviation = deviation
			origin = A
	for B in face_stickers:
		#linear equation B-A = p*a+q*b
		S = B.center-origin.center
		q = (S[1]-S[0]*a[1]/a[0])/(-b[0]*a[1]/a[0]+b[1])
		p = S[0]/a[0]-q*b[0]/a[0]
		B.p,B.q = round(p),round(q)
		print B.p,B.q
		cv2.circle(steny,tuple(map(int,origin.center+B.p*a+B.q*b)),6,100)
	mostinside = 0
	offsets=[]
	print origin.center
	for offsetp in range(-2,3): #fitting best square 3x3
		for offsetq in range(-2,3):
			print offsetp, offsetq
			inside =0
			for B in face_stickers:
				if abs(B.p-offsetp)<=1 and abs(B.q-offsetq)<=1:
					inside += 1
			print inside
			if inside == mostinside:
				offsets.append((offsetp,offsetq))
			elif inside>mostinside:
				mostinside=inside
				offsets = [(offsetp,offsetq)]
	for i in range(-10,10):
		for j in range(-10,10):
			cv2.circle(steny,tuple(map(int,origin.center+i*a+j*b)),2,255)
	ret = []
	for i in range(len(offsets)):	
		M = origin.center+(offsets[i][0]*a+offsets[i][1]*b)
		face = [M-1.5*a-1.5*b,M-1.5*a+1.5*b,M+1.5*a+1.5*b,M+1.5*a-1.5*b]
		for i in range (4):
			face[i]=int(round(face[i][0])),int(round(face[i][1]))
		ret.append(face)
	return ret


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
klastry1 = h & 0
klastry2 = h & 0
klastry3 = h & 0
klastry4 = h & 0
klastry5 = h & 0
klastry6 = h & 0
cisla = h & 0

print "start"
starttime=clock()*1000

#DETECT STICKERS
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

for i in range(len(stickers)):
	stickers[i].id = i
stickertime=clock()*1000
print "stickers detected in " + str(int(stickertime-starttime)) + " ms"

#CLUSTERING

D=[[distance(st1,st2) for st1 in stickers] for st2 in stickers]
tree = Pycluster.treecluster(distancematrix=D)

cluster_count = 1
while True:
	clusters = tree.cut(cluster_count)
	for i in range(len(stickers)): #Debug
		if (stickers[i].weight>1 and cluster_count<7):
			cv2.drawContours(eval("klastry"+str(cluster_count)), np.array([stickers[i].V]), 0, 255*(clusters[i]+1)/(cluster_count+1),-1)
	weights=[0 for i in range(cluster_count+1)]
	for i in range(len(stickers)):
		weights[clusters[i]] += stickers[i].size + stickers[i].weight
	maxcluster_weight = 0
	maxcluster_id = 0
	maximum = 0
	for i in range(cluster_count):
		if weights[i]>maximum:
			maxcluster_id = i
			maximum=weights[i]
	maxcluster_count=list(clusters).count(maxcluster_id)
	if maxcluster_count<10 and maxdistance(maxcluster_id)<=17:
		print range(len(stickers))
		print list(clusters)
		print [int(i.weight) for i in stickers]
		print [int(i.size) for i in stickers]
		print weights
		
		break
	cluster_count+= 1

clustertime=clock()*1000
print "clusters detected in " + str(int(clustertime-stickertime)) + " ms"

#FIRST FACE

firstface = find_face(maxcluster_id) #more options
cv2.fillConvexPoly(steny,np.array(firstface[0]),100)

	








#OUTPUT

for i in range(len(stickers)):
	if (stickers[i].weight>1) and clusters[i] == maxcluster_id:
		cv2.drawContours(steny, np.array([stickers[i].V]), 0, 255,-1)
	cv2.putText(cisla, str(i), (stickers[i].center[0],stickers[i].center[1]), cv2.FONT_HERSHEY_PLAIN,1,255)

for st in stickers:
	cv2.drawContours(weighted, np.array([st.V]), 0, 10*st.weight,-1)


print "done"

pictures = ['h', 's','v', 'green', 'blue', 'orange', 'yellow', 'white', 'red', 'borders', 'tetragons', 'cisla', 'weighted', 'klastry1','klastry2','klastry3','klastry4','klastry5','steny']

details = True
if (details):
	for i in xrange(19):
		plt.subplot(4,5,i+1),plt.imshow(eval(pictures[i]),'gray')
		plt.axis("off")
		plt.title(pictures[i])
	imgRGB=mpimg.imread(sys.argv[1])
	plt.subplot(4,5,20),plt.imshow(imgRGB);
else:
	plt.subplot(1,2,1),plt.imshow(steny,'gray')
	plt.axis("off")
	imgRGB=mpimg.imread(sys.argv[1])
	plt.subplot(1,2,2),plt.imshow(imgRGB);


plt.axis("off")
plt.show()
