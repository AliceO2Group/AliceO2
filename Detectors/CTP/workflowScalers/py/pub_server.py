import zmq
import random
import sys
import time

port = "500901"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)
while True:
    topic = random.randrange(0,2**32)
    #messagedata = random.randrange(1,215) - 80
    header = "CTP"
    messagedata = str(topic)+"1 2 3 4 5"
    print("Sending:",header, messagedata)
    data = str(messagedata).encode()
    header = str(header).encode()
    socket.send(memoryview(header),zmq.SNDMORE)
    socket.send(memoryview(data))
    time.sleep(1)
