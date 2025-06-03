import os
import re
import shutil

datadirs = ["data/onpu-Etou/","data/onpu-Matu/","data/onpu-Mizu/","data/onpu-Sai/","data/onpu-Sima/","data/onpu-Uno/",
            ]

outdir = "data/outdir/"
outvaldir = "data/outvaldir/"
message=[]
if (os.path.exists(outdir)):
    message.append("outdir"+ outdir+ " was already exist.")
else:
    os.makedirs(outdir)
if (not outdir.endswith("/")):
    outdir = outdir + "/"
if (os.path.exists(outvaldir)):
    message.append("outvaldir"+ outvaldir+ " was already exist.")
else:
    os.makedirs(outvaldir)
if (not outvaldir.endswith("/")):
    outvaldir = outvaldir + "/"

print(datadirs)
classtxts = []
newi = 0
basei = 0
classtable = {}
nout=0
nval=0
with open(outdir + "classes.txt", mode='w') as fw:
    for datadir in datadirs:
        basei = newi
        if (not datadir.endswith("/")):
            datadir = datadir + "/"
        print(datadir)
        filelist = os.listdir(datadir)
        filelist.sort()  # jpg->txt

        classtxt = 0
        rtable = {}
        with open(datadir + "classes.txt") as f:
            skipi = 0
            for i, line in enumerate(f):
                if (line.strip() == "DELETE" or line.strip() == "DELETED"):
                    print("SKIP DELETE")
                    skipi += 1
                else:
                    if (line.strip() in classtable):
                        rtable[str(i)] = classtable[line.strip()]
                        print("rtable", rtable)
                        skipi += 1
                    else:
                        rtable[str(i)] = i + basei - skipi
                        print("rtable", rtable)
                        print(i, line.strip(), "=>", i + basei - skipi)
                        fw.write(line)
                        classtable[line.strip()] = i + basei - skipi
                        print("table", classtable)
                        newi += 1
        for i, file in enumerate(filelist):
            if (file == "classes.txt"):  # skip
                pass
            else:
                if (file.endswith(".jpg")):
                    txt = re.sub('.jpg$', '', file)
                elif (file.endswith(".JPG")):
                    txt = re.sub('.JPG$', '', file)
                elif (file.endswith(".txt")):
                    jpg = re.sub('.txt$', '', file)
                    if (txt == jpg):
                        #print(datadir + file)
                        with open(datadir + file) as ft:
                            for ll in ft:
                                n, xy = ll.strip().split(" ", 1)
                                #print(datadir.replace("/", "_") + txt, rtable[ll.strip().split(" ", 1)[0]], xy)
                                if (i % 20 == 1 or i % 20 == 2):
                                    #print("val", i)
                                    nval+=1
                                    shutil.copyfile(datadir + txt + ".jpg",
                                                    outvaldir + datadir.replace("/", "_") + txt + ".jpg")
                                    with open(outvaldir + datadir.replace("/", "_") + txt + ".txt", mode='a') as ftw:
                                        ftw.write(str(rtable[ll.strip().split(" ", 1)[0]]) + " " + str(xy) + "\n")
                                else:
                                    #print("out", i)
                                    nout+=1
                                    shutil.copyfile(datadir + txt + ".jpg",
                                                    outdir + datadir.replace("/", "_") + txt + ".jpg")
                                    with open(outdir + datadir.replace("/", "_") + txt + ".txt", mode='a') as ftw:
                                        ftw.write(str(rtable[ll.strip().split(" ", 1)[0]]) + " " + str(xy) + "\n")


                    else:
                        print("mismatch", txt, jpg)
shutil.copyfile(outdir + "classes.txt", outvaldir + "classes.txt")
print(classtable)
print("train",nout,"files, val",nval,"files.")
if(len(message)>0):
    print("ATTENTION:", *message)
