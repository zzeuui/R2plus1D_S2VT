import csv
import os
import shutil

"""
this code for moving videos that got new caption from all videos in MSVD
"""

all_videos_path = '/home/user/dataset/MSVD/videos'
new_videos_path = '/home/iccas/r2+1d_s2vt/data/new_videos'

f = open('video_corpus.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

name_list = []
i = 0
for line in rdr:
	i = i+1
	neww = line[0]+'_'+line[1]+'_'+line[2]+'.avi'
	if i != 1 and line[0] != '' and line[0] !=' ' and not neww in name_list:
		name_list.append(neww)
		file_path = os.path.join(all_vidoes_path, neww)

		shutil.copy(file_path, new_videos_path)
f.close()
print(len(name_list))
