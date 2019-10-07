// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/RawEncoder.h"

#include "MCHMappingInterface/Segmentation.h"

#include <boost/interprocess/allocators/allocator.hpp>

using namespace o2::mch;

namespace
{

  std::map<int, int> createDEMap()
{
  std::map<int, int> m;
  int i{0};
  o2::mch::mapping::forEachDetectionElement([&m, &i](int deid) {
    m[deid] = i++;
  });
  return m;
}

int deId2deIndex(int detElemId)
{
  static std::map<int, int> m = createDEMap();
  return m[detElemId];
}

std::vector<o2::mch::mapping::Segmentation> createSegmentations()
{
  std::vector<o2::mch::mapping::Segmentation> segs;
  o2::mch::mapping::forEachDetectionElement([&segs](int deid) {
    segs.emplace_back(deid);
  });
  return segs;
}
  
const o2::mch::mapping::Segmentation& segmentation(int detElemId)
{
  static auto segs = createSegmentations();
  return segs[deId2deIndex(detElemId)];
  }

} // namespace

RawEncoder::RawEncoder(int) {}

void RawEncoder::init()
{
}
//______________________________________________________________________
void RawEncoder::process(const std::vector<Digit> digits, std::vector<uint16_t>& raw){

  raw.clear();


  for (auto& digit : digits ){
    processDigit(digit, raw);
  }

  printRawData(raw);

}
//______________________________________________________________________
int RawEncoder::processDigit(const Digit& digit, std::vector<uint16_t>& raw){

  //time: check TimStamp class
  int detID = digit.getDetID();
  int padid = digit.getPadID();
  double adc = digit.getADC();
  auto time = digit.getTimeStamp();
  int sigtime = timeConvert(time);
  
  auto& seg = segmentation(detID);


  //header information
  char hammingparitybit = 0;//to be done
  char pkt = 0; //set below
  uint16_t datalength=0;//first part of payload
  int dualSampaID =  seg.padDualSampaId(padid); //per detector element
  //
  
  
  int padDualSampaChannel = seg.padDualSampaChannel(padid);//Channel id is still Manu
  char addressChip =padDualSampaId;//tbc, could be wrong
  char addressChannel =padDualSampaChannel;
  uint32_t bx =0; //keep at 0 for the time being
  char pbitpay =0;
  
  //  uint16_t datalength=0;//first part of payload// two times appearing? to be checked, if not, there are only 80 bits, not 90
  
  //int: at least 16 bits
  
  //header format:
  // 6 bits Hamming
  // 1 bit Parity (odd) of header
  //write one char for Hamming and parity (one bit wasted)
  //3 bit Packet type coding (PKT),
  //PKT encodes if data/no data if Num words
  //trigger too early, heartbeat, data truncated
  //or sync packet or trigger too early and data truncated
  //one char for PKT?
  // 10 bits number of 10 bit words in data payload
  //two char for this 10 bit numbers
  // 4 bits Hardware address of chip
  //one char for this
  // 5 bits channel address
  //one char for channel address
  // 20 bits BX count
  // 1 bit parity (odd) of data payload
  //easiest to do a model for the actual data
  //construct everything else afterwards
  //i.e. construct Hardware address of chip
  // chnnel address
  // BX counts
  //and payload
  // in total 50 bits in header 
  //
  
  //fill
  //payload:
  int adcsum = (int) adc;//conversion according to total number

  //todo hammingparitybit construction
//dummy only for non-empty data
  //data, keep 0s for bits not needed
  pkt= 1+4;//PKT[0] and PKT[2]                                                                       
  uint16_t bits_1 = (hammingparitybit << 9)/*7 bits*/ + (pkt<<6)/*3 bits*/ + (datalength>>4)/*first 6 of 10*/ ;
  raw.emplace_back(bits_1);
  
  uint16_t bits_2 = (datalength<<12)/*2nd 4 out of 10 bits*/+ (addressChip<<8)/*4 bits*/+ (addressChannel<<3)/*5 bits*/ + (bx>>17)/*first 3 bits out of 20*/;
  raw.emplace_back(bits_2);
  uint16_t bits_3 = (bx<<1);/*second 16 bits out of 20*/
  raw.emplace_back(bits_3);
  uint16_t bits_4 = (bx<<15)/*third 1 bit out of 20*/+(pbitpay<<14)/*1 bit*/+(datalength<<13)/*tbc 10 bits*/+(sigtime>>6)/*4 bits out of 10*/;
  raw.emplace_back(bits_4);
  uint16_t bits_5 = (sigtime<<10)/*2nd 6 bits out of 10*/+(adcsum>>10)/*first 10 bits out of 20*/  ;
  raw.emplace_back(bits_5);
  uint16_t bits_6 = (adcsum<<6)/*2nd 10 bits out of 20*/;
  raw.emplace_back(bits_6);
  return raw.size();

}
//______________________________________________________________________
int RawEncoder::printRawData(std::vector<uint16_t>& raw){

  //Todo print raw data
  
  return raw.size();
}
//______________________________________________________________________
int RawEncoder::timeConvert(double time){

  //to be fixed
  //need to "subtract" bx time...
  //need to be in 10 bits
  uint16_t inttime = (int) time;
  inttime = inttime&(0<<11)&(0<<12)&(0<<13)&(0<<14)&(0<<15);
  
  return inttime;

}
°//______________________________________________________________________
bool RawEncoder::getFEEcoordfromDetector(int padid, uint16_t &cruID, uint16_t& linkID, uint16_t& dualsampaIDfee, uint16_t& padDualSampaChannelfee, std::string mapfile){

  std::ifstream file;
  file.open(mapfile);
  if(!file)
    {
      std::cerr << "Can't open file " << mapFile <<std::endl;
      return false;
    }
  uint16_t cruid, linkid, duals, channel;
  int counter = 0;
  while(!file.eof()){
    file >> cruid >> linkid >> duals >> channel;
    ++counter;
    if(counter != padid) continue;
    cruID = cruid;
    linkID = linkid;
    dualsampaIDfee = duals;
    padDualSampaChannelfee = channel;
  }

  return true;

}



