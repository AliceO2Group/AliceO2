import zmq
import random
import sys
import time

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port)
while True:
    topic = random.randrange(9999,10005)
    messagedata = random.randrange(1,215) - 80
    print("topic %d %d" % (topic, messagedata))
    header = str("CTP").encode()
    data = str(messagedata).encode()
    socket.send(memoryview(header),zmq.SNDMORE)
    socket.send(memoryview(data))
    time.sleep(1)
