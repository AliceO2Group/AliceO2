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
void RawEncoder::process(const std::vector<Digit> digits, std::vector<char>& raw){

  raw.clear();


  for (auto& digit : digits ){
    processDigit(digit, raw);
  }


}
//______________________________________________________________________
int RawEncoder::processDigit(const Digit& digit, std::vector<char>& raw){

  //time: check TimStamp class
  int detID = digit.getDetID();
  int padid = digit.getPadID();
  double adc = digit.getADC();
  double time = digit.getTimeStamp();//dataformat T: double cast ok?

  auto& seg = segmentation(detID);

  int padDualSampaId =  seg.padDualSampaId(padid);
  int padDualSampaChannel = seg.padDualSampaChannel(padid);
  
  char hammingparitybit = 0;
  char pkt = 0;
  char datalength1=0;
  char datalength2=0;
  char addressChip =padDualSampaId ;//tbc, could be wrong
  char addressChannel =padDualSampaChannel;
  char bx1 =0; //keep at 0 for the time being
  char bx2 =0; //
  char bx3=0; //
  char pbitpay =0;
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
  int timebins = timeBins();//todo generate from a distribution
  int adcsum = (int) adc;//conversion according to total number

  int adcbins[timebins];
  for (int i=0; i< timebins; ++i)
      adcbins[i] = intSignal(adcsum, timebins, i); //to be done with function

  //todo hammingparitybit construction
  
  raw.emplace_back(hammingparitybit);
  //dummy only for non-empty data
  //data, keep 0s for bits not needed
  pkt= 1+4;//PKT[0] and PKT[2]
  raw.emplace_back(pkt);
  datalength1=(timebins+1)/(pow(2,5));//+1 for signal time
  datalength2=(timebins+1)-datalength1;
  
  raw.emplace_back(datalength1);
  raw.emplace_back(datalength2);

  //dual sampa address goes from 0 to 39
  //address of channel within sampa goes from 0 to 63
  raw.emplace_back(addressChip);
  raw.emplace_back(addressChannel);
    
  for(int i=0; i< timebins; ++i)
  raw.emplace_back(adcbins[i]);
  
  //header
  //
  //to do: functional shape for a given sum of adc
  // need to sample first from a function the distribution of
  //time duration of a signal, for the first trial, just one signal

  //second step: for a given time duration (e.g. bin number) and a given ADC
  //sum, generate counts from a distribution.
  //easiest: have a function with these two parameters, that is integrated
  //for each bin,
  // these two parameters
  
  
  return raw.size();

}
//______________________________________________________________________
int RawEncoder::intSignal(int adcsum, int timebins, int timebin){

  //uniform distribution as dummy, time independent
  return adcsum/timebins;

}
//______________________________________________________________________
int RawEncoder::timeBins(){

  return 20;
}




