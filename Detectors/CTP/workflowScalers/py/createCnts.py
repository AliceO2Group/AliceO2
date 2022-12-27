#!/usr/bin/env python3
import zmq
import random
import sys
import time
port = "500901"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)
time.sleep(1)
#
run = "527345"
path = "/home/rl/counters/"
filename = path + "20221014.cc"
filecfg = path + run + ".rcfg"
# CTP Config
def sendctpconfig(starttime):
  print("starttime:",starttime)
  fcfg = open(filecfg,"r")
  lines = fcfg.readlines()
  ctpcfg = starttime+" "
  for line in lines:
    ctpcfg += line
  fcfg.close()
  print(ctpcfg)
  senddata("ctpconfig",ctpcfg)
def senddata(header, messagedata):
  global socket
  data = messagedata
  if len(data) > 20:
    data = data[0:20]
  print("Sending:",header, data)
  data = str(messagedata).encode('UTF-8')
  header = str(header).encode('UTF-8')
  msg = [header, data]
  socket.send_multipart(msg)
  time.sleep(1)
##########################
def getRunCounters():
  print("searching for run:",run," in ",filename)
  #
  f = open(filename,"r")
  #
  n = 0
  start = time.time()
  runcnts = []
  runactive = 0
  while True:
    line = f.readline()
    if not line:
      break
    items = line.split(" ")
    runfound = 0
    for i in range(1,17):
      if items[i] == run:
        runcnts.append(line)
        runfound = 1
        print("1:",line[0:20])
        break;
    if (runfound == 1) and (runactive == 0):
      runactive = 1
    if (runfound == 0) and (runactive==1):
      runcnts.append(line)
      runactive = 0
      print("0:",line[0:20])
  print("runcnts size:", len(runcnts))
  return runcnts
def replyScalers():
  runcnts = getRunCounters()
  starttime = runcnts[0].split(" ",1)[0]
  #
  sendctpconfig(starttime)
  time.sleep(5)
  senddata("sox",runcnts[0])
  for line in runcnts[1:-1]:
    senddata("ctpd",line)
  senddata("eox",runcnts[-1])
def getIndexesForScalers(scalers):
  fileinpnames = "scalersCTP_v2.txt"
  f = open(fileinpnames,"r")
  lines = []
  while True:
    line = f.readline()
    l=line[8:].split('"')
    #print(l)
    lines.append(l[0])
    if not line:
      break
  #for l in lines: print(l)
  indeces = []
  for s in scalers:
    i = lines.index(s)
    print(i,lines[i])
    indeces.append(i)
  return indeces
def getInputs(run):
  global path,filename
  dd=1
  filename = path+"20221118.cc"
  #filename = path+"20221119.cc"
  rcnts = getRunCounters()
  inputs = ['inp3','inp4','inp5','inp25','inp26']
  indeces = getIndexesForScalers(inputs)
  inpcnts = {}
  overflow = {}
  scals0 = rcnts[0].split()
  #print(scals0[indeces[0]])
  #print(rcnts[0])
  #return
  for i in indeces:
    inpcnts[i] = [int(scals0[i+dd])] # +1 due to time
    overflow[i] = 0
  for ii in range(1,len(rcnts)):
    for i in indeces:
      scals = rcnts[ii].split()
      scal = int(scals[i+dd])
      if inpcnts[i][ii-1] > scal:
        overflow[i] += 1
        print("overflow in ",i, " at ",ii)
      inpcnts[i].append(scal +overflow[i])
  for i in indeces:
    print(i,":",inpcnts[i][len(inpcnts[i])-1]-inpcnts[i][0])
  #print(inpcnts)
if __name__ == "__main__":
  #
  if len(sys.argv) == 2:
    run =  sys.argv[1]
    #int(run)
    print("run:",run)
  elif len(sys.argv) == 3:
    port =  sys.argv[1]
    int(port)
    print("port:", port)
    print("run:", run)
  #
  #replyScalers()
  getInputs(run)

