
import os

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
################################ Main controler ################################
def _main():

    funcdict = {
                'filter_train_test'     : filterTrainTest,
                'matching_outputs'      : matchingOutput}

    conf_f  = u_getPath('subway.json')
    confs   = json.load(open(conf_f))

    #...........................................................................
    funcdict[confs['op_type']](confs[confs['op_type']])
    
   
################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()
