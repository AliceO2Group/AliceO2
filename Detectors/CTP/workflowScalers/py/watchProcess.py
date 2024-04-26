#!/usr/bin/python3
import requests,re
import os,sys,time,math
import time
from subprocess import Popen, PIPE
def send2(text = "ahoj"):
  headers = { 'Content-Type':'application/json',}
  data = {
    'text':text,
    'username':'alicetrg',
  }
  response = requests.post('https://mattermost.web.cern.ch/hooks/75949oimoinr9b47uhp8c1oomh',headers=headers,json=data)
  print(response)
#
def send():
  """
     does not work always return can not parse data
  """
  Hpar = '\\"Content-Type:application/json\\"'
  texth = '\\"text\\"'
  text = '\\"lalal\\"'
  usernameh = '\\"username\\"'
  username = '\\"alicetrg\\"'
  dpar = '{}:{},{}:{}'.format(texth,text,usernameh,username)
  dpar = "{"+dpar+"}"
  dpar = '"'+dpar+'"'
  #print(dpar)
  cmd = 'curl -g -i -X POST -H {} -d {} https://mattermost.web.cern.ch/hooks/75949oimoinr9b47uhp8c1oomh'.format(Hpar,dpar)
  #cmd = 'curl -i -X POST -H {} https://mattermost.web.cern.ch/hooks/75949oimoinr9b47uhp8c1oomh'.format(Hpar)
  #cmd='curl -s -o /dev/null {} -F blob=@datafile.root {}/CTP/Calib/{}/{}/{}'.format(wpar, ccdb, actid, tval_s, tval_e)
  process = Popen(cmd.split(), stdout = PIPE, stderr = PIPE)
  stdo, stde = process.communicate()
  print(cmd)
  print(stdo)
  print("====")
  print(stde)
def getLog(service):
  #cmd = "journalctl --no-hostname --user-unit "+service
  cmd = 'journalctl --no-hostname --user-unit --since "1 day ago" '+service
  print(cmd,cmd.split())
  process = Popen(cmd.split(), stdout = PIPE, stderr = PIPE)
  stdo, stde= process.communicate()
  stdo_str = stdo.decode("utf-8")
  if stdo_str.find("No entries") != -1:
    print("No entries for service:",service)
    return None
  #print(stdo)
  return stdo_str
NMAX = 3
def parseLog(log):
  nsent = 0
  lines = log.split('\n')
  print("# lines:",len(lines))
  print(lines[len(lines)-2])
  for line in lines:
    if line.find("ERROR") != -1:
     print(line)
     if nsent < NMAX:
      send2(line)
      nsent += 1
    if line.find("ALARM") != -1:
     print(line)
     if nsent < NMAX:
      send2(line)
      nsent += 1
#
nWarn = 0
nAlarm = 0
nError = 0
def printNew(list,n,send = 0):
  print(n,len(list))
  sendnow = 0
  if len(list) > n :
    sendnow = 1
    for i in list[n:]:
      print(i)
    n = len(list)
  if sendnow and send:
    print("sending to mm:",line)
    send2(line)
  return n
def getLogFile():
  global nWarn, nAlarm,nError
  nWarnList = []
  nAlarmList = []
  nErrorList = []
  MAX = 3
  file = "/home/rl/WORK/ctpproxy110424.log"
  f = open(file,"r")
  nsent = 0
  for line in f:
    if line.find("ERROR") != -1:
      print(line)
    if line.find("ALARM") != -1:
      #print(line)
      nAlarmList.append(line)
    #send2(line)
    #nsent += 1
    if line.find("WARN") != -1:
      #print(line)
      #items = line.split("\]\[")
      #items = re.split('\[|\]',line)
      nWarnList.append(line)
  #print(nWarn)
  nWarn = printNew(nWarnList,nWarn)
  nAlarm = printNew(nAlarmList,nAlarm,1)
  f.close()
if __name__ =="__main__":
  #send2("uj0");
  while 1:
    now = time.localtime()
    current_time = time.strftime("%H:%M:%S",now)
    print("===> Time =", current_time)
    log = getLogFile()
    time.sleep(15)

  #parseLog(log)
