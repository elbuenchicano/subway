gt = [0, 0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,1,1]
gt_vec =  []
if gt[0] == 1:
    ini = 0
for i in range(1, len_files):
	if gt[i-1] == 0 and  gt[i] == 1: 
		ini = i
    if gt[i-1] == 1 and gt[i] == 0:
		gt_vec.append((ini, i))

if gt[len_files - 1] == 1:
	gt_vec.append((ini, len_files-1))