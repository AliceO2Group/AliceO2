import sys
import zmq

port = "500901"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)

if len(sys.argv) > 2:
    port1 =  sys.argv[2]
    int(port1)

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print("Collecting updates from CTP server, port:",port)
socket.connect ("tcp://localhost:%s" % port)

if len(sys.argv) > 2:
    socket.connect ("tcp://localhost:%s" % port1)
# Subscribe to zipcode, default is NYC, 10001
topicfilter = ""
socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

# Process 5 updates
total_value = 0
while(1):
    string = socket.recv_multipart()
    h = string[0].decode('UTF-8')
    data = string[1]
    if len(data) > 20:
      data = data[0:20]
    print("string:",h,data)
    #topic, messagedata = string.split()
    #total_value += int(messagedata)
    #print(topic, messagedata)

print("Average messagedata value for topic '%s' was %dF" % (topicfilter, total_value / update_nbr))

