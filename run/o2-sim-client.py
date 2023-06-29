#!/usr/bin/env python3

'''
Copyright 2019-2020 CERN and copyright holders of ALICE O2.
See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
All rights not expressly granted are reserved.

This software is distributed under the terms of the GNU General Public
License v3 (GPL Version 3), copied verbatim in the file "COPYING".

In applying this license CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization
or submit itself to any jurisdiction.
'''

# @author Sandro Wenzel

#
#  A python client that can talk to a daemonized o2-sim simulation instance.
#  Use to start new simulation runs or to control (stop) the service.
#

import time
import zmq
import psutil
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Control client talking to o2-sim service.')
parser.add_argument('--startup', help='Startup simulation service')
parser.add_argument('--command', help='Command to be submitted to a running service (see SimReconfig class).')
parser.add_argument('--block', action='store_true', help='Blocks until (simulation/action) done.')
parser.add_argument('--pid', help='pid of simulation service (autodetermined if not given).')
parser.add_argument('--service-outfile', default='o2sim-service.log', help='Logfile for the backgrounded sim-service '
                                                                           '(if started with --startup)')
args = parser.parse_args()

service_pid = None

if args.pid:
   service_pid = args.pid

if args.startup:
    commandtokens=args.startup.split()
    commandtokens+="--asservice 1".split()
    commandtokens=['o2-sim'] + commandtokens
    # if we do valgrind
    # commandtokens=['valgrind','--tool=helgrind','--trace-children=yes'] + commandtokens
    with open("simservice.out","wb") as out, open("simservice.err","wb") as err:
        pid = psutil.Popen(commandtokens, close_fds=True, stderr=err, stdout=out)
        service_pid = pid.pid
        print ("detached as pid", pid.pid)

process_name="o2-sim"
def getpids(name):
    pids=[]
    for proc in psutil.process_iter():
        if process_name == proc.name():
            pids.append(proc.pid)
    return pids

if service_pid == None:
  pids = getpids(process_name)
  if len(pids)!=1:
    print ("Cannot determine correct PID --> need to pass it")
    exit (1)
  else:
    service_pid = pids[0]


# check that sim process is actually alive
if not psutil.pid_exists(int(service_pid)):
   print ("Could not find simulation service with PID " + str(service_pid) + " .. exiting")
   exit (1)

controladdress="ipc:///tmp/o2sim-control-" + str(service_pid)
message = args.command
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(controladdress)

def getSubscriptionAddresses(basepid):
    # returns all IPC channels to subscribe to; basepid is the process pid of the driver o2sim process
    base = [ "o2sim-notifications", "o2sim-merger-notifications", "o2sim-primary-notifications", "o2sim-worker-notifications" ]
    return [ "ipc:///tmp/" + str(x) + '-' + str(basepid) for x in base ]

# subscribe to PUB notifications from the simulation system
incomingsocket = context.socket(zmq.SUB)
for addr in getSubscriptionAddresses(service_pid):
    incomingsocket.connect(addr)
incomingsocket.subscribe("")  # <---- very important !!

if args.command:
   time.sleep(0.1) #! -> necessary delay before sending??
   a = socket.send_string(message) # --> sending
   answer = socket.recv() # <--- receive answer
   code = int.from_bytes(answer, "little")

   if code == 0:
     print ("Submission successful.")
     if args.block:
       # blocking means we wait until the operation (simulation) is done serverside
       print ("... waiting for DONE message from server")
       batchdone = False
       while not batchdone:
         notification = incomingsocket.recv_string()  #
         print (notification)
         if re.match('O2SIM.*DONE', notification) != None:
            print ("Received DONE notification from server ... quitting", notification)
            batchdone = True
         if re.match('O2SIM.*FAILURE', notification) != None:
            print ("Service reported a failure ... unblocking this call")
            batchdone = True
            exit (1)

     exit (0)

   elif code == 1:
      print ("Server busy; Cannot take commands.")
   elif code == 2:
      print ("Command string faulty (Parsing failed on serverside).")
      exit (code)
   else:
      print ("Unknown return code " + str(code) + ".")
      exit (1)

# in case of startup we might also want to block until system is ready
if args.startup and args.block:
    # blocking means we wait until the operation (simulation) or the setup is done in all parts
    serverok = False
    workerok = False
    mergerok = False
    failure = False
    while not (serverok and workerok and mergerok):
        notification = incomingsocket.recv_string()
        print ("Received notification ", notification)
        if re.match('WORKER.*AWAITING\sINPUT', notification) != None:
            workerok = True
        if re.match('MERGER.*AWAITING\sINPUT', notification) != None:
            mergerok = True
        if re.match('PRIMSERVER.*AWAITING\sINPUT', notification) != None:
            serverok = True
        if re.match('.*O2SIM.*FAILURE.*', notification) != None:
            print ("Simservice reported failure ... exiting client")
            failure = True
            break

    if failure:
       exit (1)
    exit (0)

exit (0)
