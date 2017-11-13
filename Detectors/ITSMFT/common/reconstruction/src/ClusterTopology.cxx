#include "ITSMFTReconstruction/ClusterTopology.h"
//#include <stdlib.h>
//#include <stdio.h>

//using namespace o2::ITSMFT;

namespace o2
{
namespace ITSMFT{
void convertCluster2String(const Cluster &cluster, std::string &str)
{
  int rowSpan = cluster.getPatternRowSpan();
  int columnSpan = cluster.getPatternColSpan();
  int nBytes = (rowSpan*columnSpan)>>3;
  if(((rowSpan*columnSpan)%8)!=0) nBytes++;
  str.resize(nBytes+2,0);
  str[0]=rowSpan;
  str[1]=columnSpan;
  cluster.getPattern(&str[2],nBytes);
}

ClusterTopology::ClusterTopology() : mPattern(), mHash(0)
{
}

ClusterTopology::ClusterTopology(const std::string &str) : mHash(0)
{
  setPattern(str);
}

void ClusterTopology::setPattern(const std::string &str)
{
  int nBytes = (int)str.size();
  mPattern.resize(nBytes,0);
  memcpy(&mPattern[0],&str[0],nBytes);
  nBytes-=2;
  mHash = ((unsigned long)(hashFunction(mPattern.data(),mPattern.length())))<<32;
  if(nBytes>=4){
    mHash += ((((unsigned long)mPattern[2])<<24) + (((unsigned long)mPattern[3])<<16) + (((unsigned long)mPattern[4])<<8) + ((unsigned long)mPattern[5]));
  }
  else if(nBytes==3){
    mHash += ((((unsigned long)mPattern[2])<<24) + (((unsigned long)mPattern[3])<<16) + (((unsigned long)mPattern[4])<<8));
  }
  else if(nBytes==2){
    mHash += ((((unsigned long)mPattern[2])<<24) + (((unsigned long)mPattern[3])<<16));
  }
  else if(nBytes==1){
    mHash += ((((unsigned long)mPattern[2])<<24));
  }
  else{
    std::cout << "ERROR: no fired pixels\n";
    exit(1);
  }
}

unsigned int ClusterTopology::hashFunction(const void* key, int len)
{
  //
  //Developed from https://github.com/rurban/smhasher , function MurMur2
  //
  // 'm' and 'r' are mixing constants generated offline.
  const unsigned int m =0x5bd1e995;
  const int r = 24;
  // Initialize the hash
  unsigned int h = len^0xdeadbeef;
  // Mix 4 bytes at a time into the hash
  const unsigned char* data = (const unsigned char *)key;
  //int recIndex=0;
  while(len >= 4){
    unsigned int k = *(unsigned int*)data;
    k *= m;
    k ^= k >> r;
    k *= m;
    h *= m;
    h ^= k;
    data += 4;
    len -= 4;
  }
  // Handle the last few bytes of the input array
  switch(len){
    case 3: h ^= data[2] << 16;
    case 2: h ^= data[1] << 8;
    case 1: h ^= data[0];
    h *= m;
  };
  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.
  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;
  return h;
}

std::ostream& operator<<(std::ostream& os, const ClusterTopology& topology)
{
  int rowSpan = topology.mPattern[0];
  int columnSpan = topology.mPattern[1];
  os << "rowSpan: " << rowSpan << " columnSpan: " << columnSpan << " #bytes: " << topology.mPattern.length() << std::endl;
  unsigned char tempChar = 0;
  int s=0;
  int ic = 0;
  for (unsigned int i=2; i<topology.mPattern.length(); i++){
    tempChar = topology.mPattern[i];
    s=128; //0b10000000
    while(s>0){
      if(ic%columnSpan==0) os << "|";
      ic++;
      if((tempChar&s)!=0) os << '+';
      else os << ' ';
      s/=2;
      if(ic%columnSpan==0) os << "|" << std::endl;
      if(ic==(rowSpan*columnSpan)) break;
    }
    if(ic==(rowSpan*columnSpan)) break;
  }
  os<< std::endl;
  return os;
}
}
}
