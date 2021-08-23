#!/usr/bin/env python3
def bytes4(n=10):
 with open("ctp.raw", "rb") as f:
  byte = f.read(1)
  i = 0
  # number of bytes per line
  nbytes = 4
  ibytes = 0
  word = 0
  iword = 0
  while byte != b"":
    # Do stuff with byte.
    if ibytes == nbytes:
        print("W%012i w%02i %08x " % (i,iword,word))
        word = 0
        ibytes = 0
        iword += 1
        iword = (iword % 16)
    ibyte = int.from_bytes(byte,"little")
    word += (ibyte << 8*ibytes)
    ibytes += 1
    byte = f.read(1)
    i += 1 
    if i == n: break
def bytes16(n=10):
 with open("ctp.raw", "rb") as f:
  byte = f.read(1)
  i = 0
  # number of bytes per line
  nbytes = 4
  ibytes = 0
  word = 0
  iword = 0
  words4=[]
  nwords4 = 4
  iwords4 = 0
  irdh = 0
  nrdh = 6
  stopbit = 0
  feeid = 0
  packetoffset = 0
  offset = 0
  while byte != b"":
    # Do stuff with byte.
    if ibytes == nbytes:
        #print("W%012i w%02i %08x " % (i,iword,word))
        words4.append(word)
        iwords4 += 1
        word = 0
        ibytes = 0
        iword += 1
        iword = (iword % 16)
    ibyte = int.from_bytes(byte,"little")
    word += (ibyte << 8*ibytes)
    ibytes += 1
    byte = f.read(1)
    i += 1 
    #print("i",i, " packetoffset",packetoffset, "i-15+0ffset", i-15-offset)
    if packetoffset == (i+1-16-offset): 
      #print("changing",i,packetoffset,offset)
      offset = i-15
      irdh = 0
    #
    if iwords4 == nwords4:
        ss = ("W%012i " % (i))
        ss += (" %08x " % (words4[3]))
        ss += (" %08x " % (words4[2]))
        #ss += (" %04x " % (words4[2] & 0xffff))
        ss += (" %08x " % (words4[1]))
        ss += (" %08x " % (words4[0]))
        #print("irdh ",irdh)
        if irdh == 0: 
          irdh+=1  
          feeid = (words4[0] & 0xffff0000)>>16
          packetoffset = words4[2] & 0xffff
          print("RDH==================== FEEid:0x%x Packet offset:%i" % (feeid,packetoffset))
        elif irdh == 1:
          irdh+=1
        elif irdh == 2:  
          irdh+=1
          stopbit = words4[1] & 0xff0000
          #print(words4[1])
        elif irdh == 3:
          irdh+=1
        print(ss)
        iwords4 = 0
        words4.clear()
    if i == n: break
if __name__ == "__main__":
 # execute only if run as a script
 bytes16(0)
