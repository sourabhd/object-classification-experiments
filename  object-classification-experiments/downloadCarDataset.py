#!/usr/bin/env python

##
## @file
## @author Sourabh Daptardar <saurabh.daptardar@gmail.com>
## @version 1.0
##
## @section LICENSE
## This file is part of SerrePoggioClassifier.
## 
## SerrePoggioClassifier is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## SerrePoggioClassifier is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with SerrePoggioClassifier.  If not, see <http://www.gnu.org/licenses/>.
##
## @section DESCRIPTION
## This code has been developed in partial fulfillment of the M.Tech thesis
## "Explorations on a neurologically plausible model of image object classification"
## by Sourabh Daptardar, Y7111009, CSE, IIT Kanpur.
##
## This code implements a downloader class for downloading images from 
## Google Image Search.
##

################################################################################

import urllib
import urllib2
import simplejson
import os
import pprint
import string
import sys
from xml.dom.minidom import parseString

################################################################################

os.environ['http_proxy'] = 'http://sourabhd:sour1234@vsnlproxy.iitk.ac.in:3128'
os.environ['https_proxy'] = 'http://sourabhd:sour1234@vsnlproxy.iitk.ac.in:3128'
#os.environ['ftp_proxy'] = 'http://sourabhd:sour1234@vsnlproxy.iitk.ac.in:3128'
#os.environ['proxy_username'] = 'sourabhd'
#os.environ['proxy_password'] = 'sour1234'

#os.environ['http_proxy'] = 'http://sourabhd:sour1234@172.31.1.251:1003'
#os.environ['https_proxy'] = 'http://sourabhd:sour1234@172.31.1.251:1003'

################################################################################

class Downloader:
	""" Class for downloading image data sets from Google """	
	
	qlist = []   # List of query words
	anonymizer = "http://anonymouse.org/cgi-bin/anon-www.cgi/" # should end with slash
	googleSuggestURL = "http://google.com/complete/search"
	googleSuggestType = "toolbar"
	googleAJAXSearchURL = "http://ajax.googleapis.com/ajax/services/search/"
	googleAJAXSearchService = "images"
	googleAJAXSearchVersion = "1.0"
	googleAJAXSearchRSZ = "large"
	imageExt = "jpg"
	convertCmd = "convert"
	useAnon = False
	useSim = False
	tbnid = ''

	def __init__(self,qlst,useanonymizer=False,tbnid_=''):
		""" Constructor(Initialization) function :"""

		self.qlist = qlst
		self.qlist2 = []
		self.useAnon = useanonymizer
		if tbnid_ != '':
			self.useSim = True
			self.tbnid = tbnid_

	def getRelatedQueries(self):
		""" Get queries related to the input querires """
		
		if self.useAnon == True:
			anon = self.anonymizer
		else:
			anon = ""
	
		for i in range(len(self.qlist)):
			url = self.googleSuggestURL
			data = {}
			data['output'] = self.googleSuggestType
			data['q'] = self.qlist[i]
			data['ds'] = "i"
			urlValues = urllib.urlencode(data)
			fullURL = anon + url + "?" + urlValues
			print fullURL
			try:
				fp = urllib.urlopen(fullURL)
			except IOError:
				print "MyErrorMsg: Error fetching " + fullURL
			xmlResp = fp.read()
			doc = parseString(xmlResp)
			self.qlist2.append(self.qlist[i])
			for node in (doc.getElementsByTagName("toplevel"))[0].getElementsByTagName("CompleteSuggestion"):
				el = str(node.getElementsByTagName("suggestion")[0].getAttribute("data"))
				self.qlist2.append(el)
		
		self.qlist = self.qlist2

	
	def fetchImages(self,query,dirName,start=0,end=100):
		""" Fetches images of the input query from Google Image Search """
	    
		print "Created directory " + dirName
		try:
			if not os.path.exists(dirName):
				os.mkdir(dirName)
		except OSError:
			print "OSError occured"

		if self.googleAJAXSearchRSZ == 'small':
			incr = 4
		else:
			incr = 8

		count = 0	
		
		if self.useAnon == True:
			anon = self.anonymizer
		else:
			anon = ""

		for i in range(start,end,incr):
			qstring = {}
			qstring['v'] = self.googleAJAXSearchVersion
			qstring['q'] = query
			qstring['start'] = str(i)
			qstring['rsz'] = self.googleAJAXSearchRSZ
			if self.useSim == True:
				qstring['qtype'] = 'similar'
				qstring['tbnid'] = self.tbnid
			url = anon + self.googleAJAXSearchURL + self.googleAJAXSearchService + "?" + urllib.urlencode(qstring)
			print url
			search_results = urllib.urlopen(url)
			output = simplejson.loads(search_results.read()) ## Return type is 'dict'

			if output['responseStatus'] == 200:
				for rec in output['responseData']['results']:
					iurl = str(rec['url'])
					imageurl = string.replace(str(rec['url']),"%25","%")
					#print imageurl
					pos = imageurl.rfind('.')
					count = count + 1
					ext = imageurl[pos+1:]
					localfile = dirName + os.sep + str(count) + "." +  ext
					print "SAVE: " + iurl + " : " + localfile
					try:
						urllib.urlretrieve(imageurl,localfile)
						if ext != self.imageExt:
							localfile2 = dirName +os.sep + str(count) + "." + self.imageExt
							cmd = self.convertCmd + " " + localfile + " " + localfile2
							os.system(cmd)
							os.remove(localfile)
					except IOError:
						pass

			else:
				print "Error:: URL:" + url + " Response Code:" + str(output['responseStatus'])


	def download(self,basedir):
		""" Get all the related images from Google Image Search"""

		#self.addFrontRearSide() ## For adding front , side and rear views for cars
		#self.getRelatedQueries()
		for i in range(len(self.qlist)):
			dwnldDirName = basedir + os.sep + string.replace(self.qlist[i]," ","_")
			print dwnldDirName
			self.fetchImages(self.qlist[i],dwnldDirName)

	def addFrontRearSide(self):
		""" Add front, side and rear to queries (Specific to cars) """

		nlist = []
		for q in range(len(self.qlist)):
			r = self.qlist[q]
			nlist.append(r + ' front -left -right')
			nlist.append(r + ' rear -left -right')
			nlist.append(r + ' front +left -right')
			nlist.append(r + ' front -left +right')
			nlist.append(r + ' rear +left -right')
			nlist.append(r + ' rear -left +right')
		self.qlist = nlist


################################################################################
#d = Downloader([
#		'jaguar',
#		'lotus',
#		'alpha romeo',
#		'ambassador',
#		'asuna',
#		'autobianchi',
#		'bandini',
#		'bedford',
#		'bentley',
#		'bmw',
#		'bugatti',
#		'buick',
#		'cadillac',
#		'chevrolet',
#		'chrysler',
#		'daewoo',
#		'ferrari',
#		'fiat',
#		'ford',
#		'general motors',
#		'geo',
#		'gmc',
#		'holden',
#		'hummer',
#		'india taxi',
#		'innocenti',
#		'iveco',
#		'kia',
#		'lamborghini',
#		'lancia',
#		'maruti',
#		'maserati',
#		'mercedes',
#		'mitsubishi',
#		'nissan',
#		'oldsmobile',
#		'opel',
#		'panoz',
#		'pontiac',
#		'porsche',
#		'rolls',
#		'saab',
#		'saab',
#		'skoda',
#		'spyker',
#		'subaru',
#		'suzuki',
#		'talbot',
#		'tata',
#		'tesla'
#		'toyota',
#		'tvr',
#		'vauxhall',
#		'volkswagen',
#		'volvo',
#		'zastava'
#	])

#d = Downloader([
#		'sedan',
#		'hardtop',
#		'coupe',
#		'limousine',
#		'roadster',
#		'convertible',
#		'cabriolet',
#		'station wagon',
#		'hatchback',
#		'liftback',
#		'suv',
#		'coupe utility'])		

#d = Downloader(['fuji',
#		'shell',
#		'firestone',
#		'geico',
#		'fluke',
#		'sanyo',
#		'sega',
#		'raigad',
#		'xerox',
#		'mouse'
#		])
#d.download('/mnt/data/VisionDatasets/iitk/sourabhd/IITK_Cars_All2')

d = Downloader(['paris'],True,'Y0EvzxLP7M2zjM')
d.download('/mnt/input/iitk/sourabhd/QueryParis')
#d.download('/mnt/input/iitk/sourabhd/MiscQueries')
#d.download('/mnt/input/iitk/sourabhd/CarsMiscQueries/car_rear')
#d.download('/mnt/data/VisionDatasets/iitk/sourabhd/IITK_Cars_ByManufacturer2')
#d.download('/mnt/dataiiiii/VisionDatasets/iitk/sourabhd/IITK_Cars_ByType_FRS')
