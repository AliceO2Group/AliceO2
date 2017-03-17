#!/usr/bin/python
# a very simple python script to
# create and scale a topology/conf + runfile for a variable list of
# HeartbeatSampler + DataPublisher + ... + FLP + EPN
# author: S. Wenzel
#FIXME: decide whether to config for xterm or tmux

import json

NumFLP = 2
NumDP = NumFLP  # in principle we could have different number of data publishers
NumEPN = 4
EPNStartSocket = 6000
HeartbeatSocket = 5000
HeartbeatBaseName = "heartbeatSampler"
EPNBaseName = "epnReceiver"
SFBStartSocket = 5500
FLPStartSocket = 4000
ValidatorStartSocket = 7000
FLPBaseName = "flpSender"
DPBaseName = "DataPublisherDevice"
SFBBaseName = "subframeBuilder"
tcpbase = "tcp://127.0.0.1:"
detectors = ["TPC","ITS"]
datadescription = ["TPCCLUSTER", "ITSRAW"]

configfilename = "customconfig.json"
runscriptfilename = "customrunscript.sh"

# some potentially reusable API to construct devices
def initDevice(name):
    device = {}
    device["id"] = name
    device["channels"] = [] # empty list for device
    return device

# creates one channel; empty sockets
def initChannel(name, type, connect):
    channel = {}
    channel["name"] = name
    channel["type"] = type
    channel["method"] = connect
    channel["sockets"] = []
    return channel

def initAddress(url):
    return {"address":url}

def addChannel(device, channel):
    device["channels"].append(channel)

def addAddress(channel, address):
    channel["sockets"].append(address)

# adds a custom key:value pair to a dictionary
def addPair(element, key, value):
    element[key]=value

# code describing our devices follows

def writeOneDP(id):
    device = initDevice(DPBaseName+str(id))
    inchannel = initChannel("input","sub","connect")
    addAddress(inchannel, initAddress(tcpbase + str(HeartbeatSocket)))
    outchannel = initChannel("output","pub","bind")
    addAddress(outchannel, initAddress(tcpbase + str(SFBStartSocket + id)))
    addPair(outchannel, "sndBufSize", "10")
    addChannel(device, inchannel)
    addChannel(device, outchannel)
    return device

def writeOneSFB(id):
    device = initDevice(SFBBaseName + str(id))
    # the input channel
    inchannel = initChannel("input", "sub", "connect")
    addAddress(inchannel, initAddress(tcpbase + str(SFBStartSocket+id)))
    addPair(inchannel, "sndBufSize", "10")
    # the output channel
    outchannel = initChannel("output", "pub", "bind")
    addAddress(outchannel, initAddress(tcpbase + str(FLPStartSocket+id)))
    addPair(outchannel, "sndBufSize", "10")
    addChannel(device,inchannel)
    addChannel(device,outchannel)
    return device

def writeHBSamplerDevice():
    device = initDevice(HeartbeatBaseName)
    outchannel = initChannel("output", "pub", "bind")
    addAddress(outchannel, initAddress(tcpbase + str(HeartbeatSocket)))
    addChannel(device, outchannel)
    return device

# write timeframeValidator
def writeTFV():
    device = initDevice("timeframeValidator")
    channel = initChannel("input", "sub", "connect")
    addAddress(channel, initAddress(tcpbase + str(ValidatorStartSocket)))
    addChannel(device, channel)
    return device

# write single flp device
def writeOneFLP(id):
    device = initDevice(FLPBaseName + str(id))
    inchannel = initChannel("input","sub","connect")
    addAddress(inchannel, initAddress(tcpbase + str(FLPStartSocket+id)))
    outchannel = initChannel("output", "push", "connect")
    # write all EPN addresses
    for i in range(0,NumEPN):
        addAddress(outchannel, initAddress(tcpbase + str(EPNStartSocket+i)))
    addChannel(device, inchannel)
    addChannel(device, outchannel)
    return device

def writeOneEPN(id):
    # write a single EPN device
    device = initDevice(EPNBaseName + str(id))
    inchannel = initChannel("input", "pull", "bind")
    addAddress(inchannel, initAddress(tcpbase + str(EPNStartSocket+id)))
    # the ack channel
    ackchannel = initChannel("ack", "push", "connect")
    addAddress(ackchannel, initAddress("tcp://127.0.0.1:5990"))
    addPair(ackchannel, "ratelogging", "0")
    # the output channel (time frame validator)
    outchannel = initChannel("output", "pub", "bind")
    addAddress(outchannel, initAddress(tcpbase + str(ValidatorStartSocket)))
    addChannel(device, inchannel)
    addChannel(device, ackchannel)
    addChannel(device, outchannel)
    return device

def addFLPDevices(devicelist):
    for i in range(0,NumFLP):
        devicelist.append(writeOneDP(i))
        devicelist.append(writeOneSFB(i))
        devicelist.append(writeOneFLP(i))

def addEPNDevices(devicelist):
    for i in range(0,NumEPN):
        devicelist.append(writeOneEPN(i))

def getConf():
    conf = {}
    conf["fairMQOptions"] = {"devices" : []}
    devicelist = conf["fairMQOptions"]["devices"]
    # append all the the devices to this list
    addFLPDevices(devicelist)
    addEPNDevices(devicelist)
    devicelist.append(writeTFV())
    devicelist.append(writeHBSamplerDevice())
    return conf

def writeJSONConf():
    print "creating JOSN config " + configfilename
    with open(configfilename, 'w') as f:
        f.write(json.dumps(getConf()))

def writeRunScript():
    xtermcommand = "xterm -hold -e "
    dumpstring = []
    # treat data publishers
    for i in range(0, NumDP):
        # we might need to give information about detector to DP as well
        command = "DataPublisherDevice --id DataPublisherDevice" + str(i)
        command += " --data-description " + datadescription[i%2]
        command += " --mq-config " + configfilename + " --in-chan-name input --out-chan-name output &"
        dumpstring.append(xtermcommand + command)

    # treat sfbdevices
    for i in range(0, NumFLP):
        command = "SubframeBuilderDevice --id subframeBuilder" + str(i)
        command += " --mq-config " + configfilename + " --detector " + detectors[i%2] + " &"
        dumpstring.append(xtermcommand + command)
    # treat flpSender
    for i in range(0, NumFLP):
        flpCommand = "flpSender --id flpSender" + str(i)
        flpCommand += " --mq-config " + configfilename + " --in-chan-name input"
        flpCommand += " --out-chan-name output --num-epns " + str(NumEPN) + " --flp-index " + str(i)
        flpCommand += "&"
        dumpstring.append(xtermcommand + flpCommand)
    # treat EPN receiver
    for i in range(0, NumEPN):
        epnCommand = "epnReceiver --id epnReceiver" + str(i)
        epnCommand += " --mq-config " + configfilename + " --in-chan-name input --out-chan-name output"
        epnCommand += " --num-flps " + str(NumFLP) + "&"
        dumpstring.append(xtermcommand + epnCommand)
    # treat timeFrameValidator
    command = "TimeframeValidatorDevice --id timeframeValidator --mq-config " + configfilename + " customconfig.json --input-channel-name input &"
    dumpstring.append(xtermcommand + command)

    # treat heartbeatsampler
    command = "heartbeatSampler --id heartbeatSampler --mq-config " + configfilename + "  --out-chan-name output &"
    dumpstring.append(xtermcommand + command)

    print "creating runscript " + runscriptfilename
    with open(runscriptfilename, 'w') as f:
        f.write('\n'.join(dumpstring))

if __name__ == '__main__':
    writeJSONConf()
    writeRunScript()
