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
path = "/home/rl/countersLHCo/"
#
run = "527345"
filename = path + "20221014.cc"
#
#run = "528543"
#run = "528537"
#filename = path + "20221102.cc"
#
#run = "527963"
#filename = path + "20221024.cc"
#
#run = "527349"
#filename = path + "20221015.cc"
#
filecfg = path + run + ".rcfg"
#
if len(sys.argv) == 2:
  run =  sys.argv[1]
  int(run)
elif len(sys.argv) == 3:
  port =  sys.argv[1]
  int(port)
  print("port:", port)
print("run:", run)
#

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
  test = str("test").encode('UTF-8')
  msg = [header, data,test]
  socket.send_multipart(msg)
  time.sleep(1)
##########################
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
starttime = runcnts[0].split(" ",1)[0]
#
sendctpconfig(starttime)
time.sleep(5)
senddata("sox",runcnts[0])
for line in runcnts[1:-1]:
  senddata("ctpd",line)
senddata("eox",runcnts[-1])



