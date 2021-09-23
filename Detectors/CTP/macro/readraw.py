#!/usr/bin/env python3
import struct
import numpy as np
NGBT = 80
def makeGBTWordInverse(dglets, GBTWord, rem, s_gbt, Npld):
  diglet = rem
  i = 0
  while i<(NGBT - Npld):
    packed_bytes = struct.pack('IIH',0,0,0);
    masksize = int.from_bytes(packed_bytes,"little")
    for j in range(Npld-s_gbt): masksize |= (1 << j)
    diglet |= (GBTWord & masksize) << s_gbt
    dglets.append(diglet)
    diglet = 0
    i += Npld-s_gbt
    #print(Npld,s_gbt)
    GBTWord = (GBTWord >> (Npld - s_gbt))
    s_gbt = 0
  s_gbt = NGBT - i
  rem = GBTWord
  return dglets,rem,s_gbt
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
  orbit0 = 0
  size_gbt = 0
  packed_bytes = struct.pack('IIH',0,0,0);
  remnant = int.from_bytes(packed_bytes,"little")
  byte = f.read(1)
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
      orbit = -1
      bcid = -1
    #
    if iwords4 == nwords4:
        ss = ("W%012i " % (i))
        ss += (" %08x " % (words4[3]))
        ss += (" %08x " % (words4[2]))
        #ss += (" %04x " % (words4[2] & 0xffff))
        ss += (" %08x " % (words4[1]))
        ss += (" %08x " % (words4[0]))
        #print("irdh ",irdh)
        #format = struct.Struct('<IIH')
        w2 = int(words4[2] & 0xffff);
        packed_bytes = struct.pack('IIH',words4[0],words4[1],w2);
        uss = int.from_bytes(packed_bytes,"little")
        prtflag=0
        if irdh == 0: 
          irdh+=1  
          feeid = (words4[0] & 0xffff0000)>>16
          packetoffset = words4[2] & 0xffff
          print("RDH==================== ccccFEEid:0x%x Packet offset:%i" % (feeid,packetoffset))
        elif irdh == 1:
          irdh+=1
          bcid =  words4[0] & 0xfff;
          orbit = words4[1]
        elif irdh == 2:  
          irdh+=1
          stopbit = words4[1] & 0xff0000
          #print(words4[1])
        elif irdh == 3:
          irdh+=1
          print("RDH---: ORBIT:0x%x BCID:0x%x" % (orbit,bcid))
          if orbit0 != orbit:
            remnant = 0
            size_gbt = 0
            orbit0 = orbit
        else:
          #print("Decode")
          pp = (f'{uss:010x}')
          if len(pp) > 20: 
            print("Internal error")
            exit(1)
          elif len(pp) < 20:
            aa=""
            for ii in range(20-len(pp)): aa += "0"
            pp = aa+pp
          diglets = []
          if feeid == 0x121: diglets,remnant,size_gbt = makeGBTWordInverse(diglets,uss,remnant,size_gbt,64+12)
          else : diglets,remnant,size_gbt = makeGBTWordInverse(diglets,uss,remnant,size_gbt,48+12)
          pss="i:"+f'{i:10d}'
          flag = 0
          for d in diglets: 
            if feeid == 0x121: flag += d & 0xffffffffffffffff000
            else: flag+= d & 0xffffffffffff000
            pss += " "+f'{d:010x}'
            bcid = d & 0xfff;
            pss += f'{bcid:5d}'
            pss += " orb:"+""f'{orbit:4d}'
          #flag=1  
          if flag: flag = 123456789
          print(pp,len(pp),"pld:",pss, " flag:",flag)
          #print("payload:",pss)
          #print(ss)
          prtflag = 1
        if prtflag == 0: print(ss)
        iwords4 = 0
        words4.clear()
    if i == n: break
if __name__ == "__main__":
 # execute only if run as a script
 bytes16(0)
