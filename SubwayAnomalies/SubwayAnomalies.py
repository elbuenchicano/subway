
import os
import cv2
import numpy as np

from utils import *


################################################################################
################################################################################
def filterTrainTest(confs):
    
    trk_list    = confs['directory']
    end_train   = confs['end_train']
    token       = confs['token']  
    out         = confs['out']
    
    flist   = u_loadFileManager(trk_list, token)
    flist   = sorted(flist, key = u_stringSplitByNumbers)

    train_f     = []
    train_r     = []
    train_a     = []
    train_t     = []
    train_i     = []


    test_f      = []
    test_r      = []
    test_a      = []
    test_t      = []
    test_i      = []
    
    prog    = 0
    end     = len(flist)
    for i in flist:
        u_progress(prog, end)
        x       =  open(i, 'r')
        line    = x.readline()
        ini     = int(line.split(',')[0])

        i       = i.replace('\\', '/')

        a       = i.replace('.trk', '_mtA.png')
        r       = i.replace('.trk', '_mtR.png')
        f       = i.replace('.trk', '.ft')
        im      = i.replace('.trk', '_d3i.png')
        

        if ini < end_train:
            train_t.append(i)
            train_a.append(a)
            train_r.append(r)
            train_f.append(f)
            train_i.append(im)
        else:
            test_t.append(i)
            test_a.append(a)
            test_r.append(r)
            test_f.append(f)
            test_i.append(im)


        prog    += 1

    u_saveList2File(out+'/train_angular.lst', train_a)
    u_saveList2File(out+'/test_angular.lst', test_a)

    u_saveList2File(out+'/train_radial.lst', train_r)
    u_saveList2File(out+'/test_radial.lst', test_r)

    u_saveList2File(out+'/train_trk.lst', train_t)
    u_saveList2File(out+'/test_trk.lst', test_t)

    u_saveList2File(out+'/train_centers.lst', train_f)
    u_saveList2File(out+'/test_centers.lst', test_f)

    u_saveList2File(out+'/train_im.lst', train_i)
    u_saveList2File(out+'/test_im.lst', test_i)

################################################################################
def filterRange(confs):
    
    trk_list    = confs['directory']
    frame_ini   = confs['frame_ini']
    frame_fin   = confs['frame_fin']

    token       = confs['token']  
    out         = confs['out']
    out_token   = confs['out_token']
    
    flist   = u_loadFileManager(trk_list, token)
    flist   = sorted(flist, key = u_stringSplitByNumbers)

    train_f     = []
    train_r     = []
    train_a     = []
    train_t     = []
    train_i     = []


    u_mkdir(out)
    
    prog    = 0
    end     = len(flist)
    for i in flist:
        u_progress(prog, end)
        x       =  open(i, 'r')
        line    = x.readline()
        ini     = int(line.split(',')[0])

        i       = i.replace('\\', '/')

        a       = i.replace('.trk', '_mtA.png')
        r       = i.replace('.trk', '_mtR.png')
        f       = i.replace('.trk', '.ft')
        im      = i.replace('.trk', '_d3i.png')
        

        if ini >= frame_ini and ini <= frame_fin:
            train_t.append(i)
            train_a.append(a)
            train_r.append(r)
            train_f.append(f)
            train_i.append(im)

        prog    += 1

    u_saveList2File(out+'/'+out_token+'_angular.lst', train_a)
    u_saveList2File(out+'/'+out_token+'_radial.lst', train_r)
    u_saveList2File(out+'/'+out_token+'_trk.lst', train_t)
    u_saveList2File(out+'/'+out_token+'_centers.lst', train_f)
    u_saveList2File(out+'/'+out_token+'_im.lst', train_i)
        
################################################################################
################################################################################    
def matchingOutput(data):
    directory   = data['directory']
    token       = data['token']
    afile       = data['afile']

    out_token   = data['out_token']
    out_dir     = data['out_dir']

    u_mkdir(out_dir)

    ##.........................................................................
    files   = u_loadFileManager(directory, token)
    files   = sorted(files, key = u_stringSplitByNumbers)

    afile = u_fileList2array(afile)

    anomaly_trk = {}
    for item in afile:
        tok = item.split('_')[1]
        trk = int(tok.split('.')[0])
        anomaly_trk[trk] = ''

    out = []
    for item in files:
        trkfiles    = u_fileList2array(item)
        clusterid   = item.split('_')[-2]
        obs = []
        for fil in trkfiles:
            tok = fil.split('_')[1]
            trk = int(tok.split('.')[0])
            if trk in anomaly_trk:
                anomaly_trk[trk] = clusterid
                obs.append('1')
            else:
                obs.append('0')
        out.append(obs)

    # saving data .............................................................
    out_string = ''
    for key in sorted(anomaly_trk):
        out_string += str(key) + ' ' + anomaly_trk[key] + '\n' 

    name = out_dir + '/' + out_token + '_afile.txt'
    u_save2File(name, out_string)

    cont = 0
    out_string = ''
    for clus in out:
        out_string += 'Cluster ' + str(cont) + '\n\n' 
        for item in clus:
            out_string += item + '\n' 
        out_string += '==============================\n' 
        cont       += 1

    name = out_dir + '/' + out_token + '_clusters.txt'
    u_save2File(name, out_string)

################################################################################
################################################################################
def trainSequenceGt(data):
    gt_folder   = data['gt_folder']
    token       = data['token']
    out         = data['out']
    th_rate     = data['th_rate']
    add         = data['add']

    #...........................................................................
    files   = u_loadFileManager(gt_folder, token)
    files   = sorted(files, key = u_stringSplitByNumbers)

    first   = cv2.imread(files[0], 0)

    thr     = first.shape[0] * first.shape[1] * th_rate

    len_files   = len(files)
    gt          = np.zeros(len_files)
      
    #...........................................................................
    for i in range(len_files):
        u_progress(i, len_files)
        img = cv2.imread(files[i], 0)/255
        inf = img.sum()
        if inf >= thr:
            gt[i] =  1
    print('')

    #...........................................................................
    gt_vec =  []
    if gt[0] == 1:
        ini = 0
    for i in range(1, len_files):
        u_progress(i, len_files,'last')
        if gt[i-1] == 0 and  gt[i] == 1: 
            ini = i
        if gt[i-1] == 1 and  gt[i] == 0:
            gt_vec.append((ini + add, i-1 + add))
    print('')

    if gt[len_files - 1] == 1:
        gt_vec.append((ini + add, len_files-1 + add))

    u_saveArrayTuple2File(out, gt_vec)
    print('gt saved in:', out)
        
    

################################################################################
################################################################################    
def cameraTrainTest(data):
    directory   = data['directory']
    token       = data['token']
    dir_target  = data['dir_target']
    ext_target  = data['ext_target']
    out_dir     = data['out_dir']
    out_token   = data['out_token']
    th          = data['th_day']
    in_token    = data['in_token']

    #..........................................................................
    u_mkdir(out_dir)

    flist       = u_listFileAll(directory, token)
    flist       = sorted (flist, key = u_stringSplitByNumbers)

    featl   = []
    trkl    = []
    imgl    = []
    rad     = []
    ang     = []

    featl_t = []
    trkl_t  = []
    imgl_t  = []
    rad_t   = []
    ang_t   = []

    featl_b = []
    trkl_b  = []
    imgl_b  = []
    rad_b   = []
    ang_b   = []

    ttoken  = th.replace('.', '_')
    th      =  float(th)

    ext_target_ = '.' + ext_target 

    for file in flist:
        vfiles  = u_fileList2array(file)
        for dire in vfiles:
            dire = dire[:-4]
            sub_dire    = dir_target + '/' + dire 
            flist       = u_listFileAll(sub_dire, ext_target)

            date    = dire.split('_')[0]
            y, m, d = date.split('-')
            m_d     = float(m) + float(d)/100 
            
            if  m_d > th:
                for fil in flist:
                    
                    if fil.find(in_token) > 0:
 
                        trk_name    = fil.replace(ext_target, in_token + '.trk')
                        d3i_name    = fil.replace(ext_target_, in_token + '_d3i.png')
                        rad_name    = fil.replace(ext_target_, in_token + '_mtR.png')
                        ang_name    = fil.replace(ext_target_, in_token + '_mtA.png')

                        featl.append(fil)
                        trkl.append(trk_name)
                        imgl.append(d3i_name)
                        rad.append(rad_name)
                        ang.append(ang_name)

                        featl_t.append(fil)
                        trkl_t.append(trk_name)
                        imgl_t.append(d3i_name)
                        rad_t.append(rad_name)
                        ang_t.append(ang_name)
            else:
                for fil in flist:
                    if fil.find(in_token) > 0:

                        trk_name    = fil.replace(ext_target, in_token + '.trk')
                        d3i_name    = fil.replace(ext_target_, in_token + '_d3i.png')
                        rad_name    = fil.replace(ext_target_, in_token + '_mtR.png')
                        ang_name    = fil.replace(ext_target_, in_token + '_mtA.png')

                        featl.append(fil)
                        trkl.append(trk_name)
                        imgl.append(d3i_name)
                        rad.append(rad_name)
                        ang.append(ang_name)

                        featl_b.append(fil)
                        trkl_b.append(trk_name)
                        imgl_b.append(d3i_name)
                        rad_b.append(rad_name)
                        ang_b.append(ang_name)

    featl   = sorted(featl, key =  u_stringSplitByNumbers)
    featl_t = sorted(featl_t, key =  u_stringSplitByNumbers)
    featl_b = sorted(featl_b, key =  u_stringSplitByNumbers)
    
    trkl    = sorted(trkl, key =  u_stringSplitByNumbers)
    trkl_t  = sorted(trkl_t, key =  u_stringSplitByNumbers)
    trkl_b  = sorted(trkl_b, key =  u_stringSplitByNumbers)
    
    imgl    = sorted(imgl, key =  u_stringSplitByNumbers)
    imgl_t  = sorted(imgl_t, key =  u_stringSplitByNumbers)
    imgl_b  = sorted(imgl_b, key =  u_stringSplitByNumbers)

    rad     = sorted(rad, key =  u_stringSplitByNumbers)
    rad_t  = sorted(rad_t, key =  u_stringSplitByNumbers)
    rad_b  = sorted(rad_b, key =  u_stringSplitByNumbers)

    ang    = sorted(ang, key =  u_stringSplitByNumbers)
    ang_t  = sorted(ang_t, key =  u_stringSplitByNumbers)
    ang_b  = sorted(ang_b, key =  u_stringSplitByNumbers)
    


    # saving data...............................................................
    out_dir = out_dir + '/' 
    name    = out_dir + out_token + '_centers.lst'  
    u_saveArray2File(name, featl)

    name    = out_dir + out_token + '_trks.lst'  
    u_saveArray2File(name, trkl)

    name    = out_dir + out_token + '_imgs.lst'  
    u_saveArray2File(name, imgl)

    name    = out_dir + out_token + '_ang.lst'  
    u_saveArray2File(name, ang)

    name    = out_dir + out_token + '_rad.lst'  
    u_saveArray2File(name, rad)

    name    = out_dir + out_token + '_u_' + ttoken + '_centers.lst'  
    u_saveArray2File(name, featl_t)

    name    = out_dir + out_token + '_u_' + ttoken + '_trks.lst'  
    u_saveArray2File(name, trkl_t)

    name    = out_dir + out_token + '_u_' + ttoken + '_imgs.lst'  
    u_saveArray2File(name, imgl_t)

    name    = out_dir + out_token + '_u_' + ttoken + '_ang.lst'  
    u_saveArray2File(name, ang_t)

    name    = out_dir + out_token + '_u_' + ttoken + '_rad.lst'  
    u_saveArray2File(name, rad_t)

    name    = out_dir + out_token + '_d_' + ttoken + '_centers.lst'  
    u_saveArray2File(name, featl_b)

    name    = out_dir + out_token + '_d_' + ttoken + '_trks.lst'  
    u_saveArray2File(name, trkl_b)

    name    = out_dir + out_token + '_d_' + ttoken + '_imgs.lst'  
    u_saveArray2File(name, imgl_b)

    name    = out_dir + out_token + '_d_' + ttoken + '_ang.lst'  
    u_saveArray2File(name, ang_b)

    name    = out_dir + out_token + '_d_' + ttoken + '_rad.lst'  
    u_saveArray2File(name, rad_b)


    name    = out_dir + out_token + '_files_readed.lst'  
    u_saveArray2File(name, flist)

    
################################################################################
################################################################################    
################################ Main controler ################################
def _main():

    funcdict = {
                'filter_train_test'     : filterTrainTest,
                'matching_outputs'      : matchingOutput,
                'train_seq'             : trainSequenceGt,
                'camera_train_test'     : cameraTrainTest,
                'filter_range'          : filterRange}

    conf_f  = u_getPath('subway.json')
    confs   = json.load(open(conf_f))

    #...........................................................................
    funcdict[confs['op_type']](confs[confs['op_type']])
    
   
################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()
