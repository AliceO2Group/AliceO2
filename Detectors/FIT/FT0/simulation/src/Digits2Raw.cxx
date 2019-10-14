// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/*
  Digits to RAW data coding. RAW data format - class DataFormat/Detectors/FIT/FT0/RawEventData
  18 PMs (GBT links) 12 MCPs each  and 1 TCM, each stream transmit separately
Event header - 80bits
  uint startDescriptor : 4;
  uint nGBTWords : 4;
  uint reservedField : 32;
  uint orbit : 32;
  uint bc : 12;

  Event data 40bits
  short int time : 12;
  short int charge : 12;
  unsigned short int numberADC : 1;
  bool isDoubleEvent : 1;
  bool is1TimeLostEvent : 1;
  bool is2TimeLostEvent : 1;
  bool isADCinGate : 1;
  bool isTimeInfoLate : 1;
  bool isAmpHigh : 1;
  bool isEventInTVDC : 1;
  bool isTimeInfoLost : 1;
  uint channelID : 4;
GBT packet:
RDH + Event header + event data, 2 channels per 1 GBT word;
if no data for this PM - only headers.

Trigger mode : detector sends data to FLP at each trigger;
Continueous mode  :   for only bunches with data at least in 1 channel.  
*/

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/RawEventData.h"
#include "FT0Simulation/Digits2Raw.h"
#include "DetectorsBase/Triggers.h"
#include <Framework/Logger.h>
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <vector>
#include <bitset>
#include <iomanip>
#include "TFile.h"
#include "TTree.h"

using namespace o2::ft0;

ClassImp(Digits2Raw);
void setRDH(o2::header::RAWDataHeader&, int nlink, o2::InteractionRecord const&);
EventHeader makeGBTHeader(int link, o2::InteractionRecord const& mIntRecord);

Digits2Raw::Digits2Raw(const std::string fileRaw, std::string fileDigitsName)
{

  mFileDest.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  mFileDest.open(fileRaw, std::fstream::out | std::fstream::binary);
  Digits2Raw::readDigits(fileDigitsName.c_str());
}

void Digits2Raw::readDigits(const std::string fileDigitsName)
{
  LOG(INFO) << "**********Digits2Raw::convertDigits" << std::endl;

  o2::ft0::LookUpTable lut{o2::ft0::Digits2Raw::linear()};
  LOG(INFO) << " ##### LookUp set ";
  
  TFile* fdig = TFile::Open(fileDigitsName.data());
  assert(fdig != nullptr);
  LOG(INFO) << " Open digits file ";
  TTree* digTree = (TTree*)fdig->Get("o2sim");
  std::vector<o2::ft0::Digit>* digArr = new std::vector<o2::ft0::Digit>;
  digTree->SetBranchAddress("FT0Digit", &digArr);
  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  uint32_t old_orbit = ~0;

  for (Int_t iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
    for (const auto& digit : *digArr) {
      auto intRecord = digit.getInteractionRecord();
      uint32_t current_orbit = intRecord.orbit;
      LOG(INFO) << "!!!@@@@ old orbit " << old_orbit << " new orbit " << current_orbit;
      if (old_orbit != current_orbit) {
        for (DataPageWriter& writer : mPages)
          writer.flush(mFileDest);
        for (int nlink = 0; nlink < NPMs; ++nlink)
          setRDH(mPages[nlink].mRDH, nlink, intRecord);
        old_orbit = current_orbit;
      }
      convertDigits(digit, lut);
    }
  }
}

/*******************************************************************************************************************/
void Digits2Raw::convertDigits(const o2::ft0::Digit& digit, const o2::ft0::LookUpTable& lut)
{
  auto intRecord = digit.getInteractionRecord();
  std::vector<o2::ft0::ChannelData> mTimeAmp = digit.getChDgData();
  // check empty event
  if (mTimeAmp.size() != 0) {
    bool is0TVX = digit.getisVrtx();
    int oldlink = -1;
    int nchannels = 0;
    //   o2::ft0::RawEventData rawEventData;
    for (auto& d : mTimeAmp) {
      int nlink = lut.getLink(d.ChId); 
      if (nlink != oldlink) {  
        if (oldlink >= 0) { 
          uint nGBTWords = uint((nchannels + 1) / 2);
          if ((nchannels % 2) == 1)
            mRawEventData.mEventData[nchannels] = {};
          mRawEventData.mEventHeader.nGBTWords = nGBTWords;
          mPages[oldlink].write(mRawEventData.to_vector());
        }
        oldlink = nlink;
        mRawEventData.mEventHeader = makeGBTHeader(nlink, intRecord);
        LOG(INFO) << " mRawEventData.mEventHeader size " << sizeof(mRawEventData.mEventHeader);
        nchannels = 0;
      }
      std::cout << " ChID digits " << d.ChId << " link " << nlink << " channel " << lut.getMCP(d.ChId) << " amp " << d.QTCAmpl << " time " << d.CFDTime << std::endl;
      auto& newData = mRawEventData.mEventData[nchannels];
      newData.charge = MV_2_Nchannels * d.QTCAmpl;   //7 mV ->16channels
      newData.time = CFD_NS_2_Nchannels * d.CFDTime; //1000.(ps)/13.2(channel);
      newData.is1TimeLostEvent = 0;
      newData.is2TimeLostEvent = 0;
      newData.isADCinGate = 1; 
      newData.isAmpHigh = 0; 
      newData.isDoubleEvent = 0;
      newData.isEventInTVDC = is0TVX ? 1 : 0;
      newData.isTimeInfoLate = 0;
      newData.isTimeInfoLost = 0;
      int chain = std::rand() % 2;
      newData.numberADC = chain ? 1 : 0;
      newData.channelID = lut.getMCP(d.ChId);
      std::cout << "@@@@ packed GBT " << nlink << " channelID   " << (int)newData.channelID << " charge " << newData.charge << " time " << newData.time << " chain " << int(newData.numberADC) << " size " << sizeof(newData) << std::endl;
      nchannels++;
    }
    // fill mEventData[nchannels] with 0s to flag that this is a dummy data
    uint nGBTWords = uint((nchannels + 1) / 2);
    if ((nchannels % 2) == 1)
      mRawEventData.mEventData[nchannels] = {};
    mRawEventData.mEventHeader.nGBTWords = nGBTWords;
    mPages[oldlink].write(mRawEventData.to_vector());
  }
}

//_____________________________________________________________________________________
EventHeader makeGBTHeader(int link, o2::InteractionRecord const& mIntRecord)
{
  EventHeader mEventHeader{};
  mEventHeader.startDescriptor = 0xf;
  mEventHeader.reservedField = 0;
  mEventHeader.bc = mIntRecord.bc;
  mEventHeader.orbit = mIntRecord.orbit;
  return mEventHeader;
}
//_____________________________________________________________________________
void setRDH(o2::header::RAWDataHeader& mRDH, int nlink, o2::InteractionRecord const& mIntRecord)
{
  mRDH.triggerOrbit = mRDH.heartbeatOrbit = mIntRecord.orbit;
  mRDH.triggerBC = mRDH.heartbeatBC = mIntRecord.bc;
  mRDH.linkID = nlink;
  mRDH.feeId = nlink;

  mRDH.triggerType = o2::trigger::PhT; // ??
  mRDH.detectorField = 0xffff;         //empty for FIt yet
  mRDH.blockLength = 0xffff;           // ITS keeps this dummy
  mRDH.stop = 0;                       // ??? last package  on page

  //  mRDH.pageCnt = nPages;
  //  printRDH(&mRDH);
}
//_____________________________________________________________________________

void Digits2Raw::close()
{
  if (mFileDest.is_open())
    mFileDest.close();
}
