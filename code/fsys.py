import re
import os
import queue


# directory tools
def cd(dir):
	os.chdir(dir)


def mkdir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def mkcd(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)
	os.chdir(dir)


def getcwd():
	return os.getcwd()


# file/folder query
def file(wildcards='*', natSort=True):
	"""
	filelist = file(wildcards = '*' , natSort = True)
	returns list of files that match matlab style pattern matching
	natural sort is used by default
	i.e., file('path/to/dir/*') [all files]
	file('path/to/dir/*.end') [enforces ends with]
	file('path/to/dir/*mid*') [all files containing mid]
	file('path/to/dir/start*') [enforces starts with]
	"""
	path, fileWild = os.path.split(wildcards)
	# deal with empty cases
	if not path:
		path = '.'
	if not fileWild:
		fileWild = '*'

	# construct wildcard expression
	subpatterns = list(filter(None, re.split('\*', fileWild)))
	if fileWild[0] != '*':
		subpatterns[0] = '^' + subpatterns[0]
	if fileWild[-1] != '*':
		subpatterns[-1] = subpatterns[-1] + '$'
	subpatterns = [pat.join(['(', ')']) for pat in subpatterns]
	pattern = '(.*)'.join(subpatterns)
	filelist = []

	for f in os.listdir(path):
		if bool(re.search(pattern, f, flags=re.I)) and os.path.isfile('/'.join([path, f])) and f[0] != '.':
			if path == '.':
				filelist.append(f)
			else:
				filelist.append(path + '/' + f)
	if natSort:
		filelist = natural_sort(filelist)
	return filelist


def folder(wildcards='*', natSort=True):
	"""
	folderlist = folder(wildcards = '*' , natSort = True)
	returns list of folders that match matlab style pattern matching
	natural sort is used by default
	i.e., folder('path/to/dir/*') [all folders]
	folder('path/to/dir/*end') [enforces ends with]
	folder('path/to/dir/*mid*') [all folders containing ex]
	folder('path/to/dir/start*') [enforces starts with]
	"""
	path, folderWild = os.path.split(wildcards)
	# deal with empty cases
	if not path:
		path = '.'
	if not folderWild:
		folderWild = '*'

	# construct wildcard expression
	subpatterns = list(filter(None, re.split('\*', folderWild)))
	if folderWild[0] != '*':
		subpatterns[0] = '^' + subpatterns[0]
	if folderWild[-1] != '*':
		subpatterns[-1] = subpatterns[-1] + '$'
	subpatterns = [pat.join(['(', ')']) for pat in subpatterns]
	pattern = '(.*)'.join(subpatterns)
	folderlist = []
	for f in os.listdir(path):
		if bool(re.search(pattern, f, flags=re.I)) and os.path.isdir('/'.join([path, f])) and f[0] != '.':
			if path == '.':
				folderlist.append(f + '/')
			else:
				folderlist.append(path + '/' + f + '/')
	if natSort:
		folderlist = natural_sort(folderlist)
	return folderlist


# file/folder mating
def pair(flist1, flist2):
	if len(flist1) != len(flist2):
		raise ValueError('flist1 must be same length as flist2')
	return list(zip(flist1, flist2))


def pair_fuzzy(flist1, flist2):
	"""
	matched,unmatched1,unmatched2 = matchpair(flist1,flist2)
	provides fuzzy matching for file or folder lists using
	Levenshtein edit distance and returns list of string tuples
	"""
	# constructs list of potential matches
	que = queue.PriorityQueue()
	for f1 in flist1:
		minDist = float('Inf')
		match = []
		for f2 in flist2:
			dist = levenshteinDistance(f1, f2)
			if dist < minDist:
				match = list([f2])
				minDist = dist
			elif dist == minDist:
				match.append(f2)
		que.put((minDist, f1, match))

	# now need to check for duplicates
	unmatched1 = flist1
	unmatched2 = flist2
	matched1 = []
	matched2 = []
	while not que.empty():
		dist, f1, f2List1 = que.get()
		f2List2 = [f for f in f2List1 if f not in matched2]
		if (len(f2List2) == 1) and (f1 in unmatched1) and (f2List2[0] in unmatched2):
			matched1.append(f1)
			matched2.append(f2List2[0])
			unmatched1 = [f for f in unmatched1 if f != f1]
			unmatched2 = [f for f in unmatched2 if f != f2List2[0]]
		elif len(f2List2) < len(f2List1):
			if f2List2:
				que.put((dist, f1, f2List2))
	return list(zip(matched1, matched2)), unmatched1, unmatched2


def pair_offset(flist1, flist2, offset=0):
	matched = []
	for i1 in range(len(flist1)):
		i2 = i1 + offset
		if (i2 >= 0) and (i2 <= (len(flist2) - 1)):
			matched.append((flist1[i1], flist2[i2]))
	return matched


# helper functions...
def file_parts(file):
	pathToFile, ext = os.path.splitext(file)
	path, file = os.path.split(pathToFile)
	return path, file, ext


def natural_sort(l):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key=alphanum_key)


def levenshteinDistance(s1, s2):
	s1 = s1.lower()
	s2 = s2.lower()
	if len(s1) > len(s2):
		s1, s2 = s2, s1

	distances = range(len(s1) + 1)
	for i2, c2 in enumerate(s2):
		distances_ = [i2 + 1]
		for i1, c1 in enumerate(s1):
			if c1 == c2:
				distances_.append(distances[i1])
			else:
				distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
		distances = distances_
	return distances[-1]

# os.chdir('/Users/colin/Dropbox/__Atlas__/data/30890')
# hist = file('histology/*')
# masks = file('masks/*')
# matched,unmatched1,unmatched2 = matchpair(hist,masks)
# print(matched)
# print(' ')
# print(unmatched1)
# print(' ')
# print(unmatched2)
# print(' ')
