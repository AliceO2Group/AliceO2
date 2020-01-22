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
#include "CommonConstants/Triggers.h"
#include "CommonUtils/HBFUtils.h"
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
  LOG(INFO) << " Open digits file " << fileDigitsName.data();
  TTree* digTree = (TTree*)fdig->Get("o2sim");

  std::vector<o2::ft0::Digit> ft0BCData, *ft0BCDataPtr = &ft0BCData;
  std::vector<o2::ft0::ChannelData> ft0ChData, *ft0ChDataPtr = &ft0ChData;

  digTree->SetBranchAddress("FT0DigitBC", &ft0BCDataPtr);
  digTree->SetBranchAddress("FT0DigitCh", &ft0ChDataPtr);

  uint32_t old_orbit = ~0;
  o2::InteractionRecord intRecord;
  o2::InteractionRecord lastIR = mSampler.getFirstIR();
  std::vector<o2::InteractionRecord> HBIRVec;

  for (int ient = 0; ient < digTree->GetEntries(); ient++) {
    digTree->GetEntry(ient);

    int nbc = ft0BCData.size();
    LOG(INFO) << "Entry " << ient << " : " << nbc << " BCs stored";
    int itrig = 0;

    for (int ibc = 0; ibc < nbc; ibc++) {
      const auto& bcd = ft0BCData[ibc];
      bcd.print();
      intRecord = bcd.getInteractionRecord();
   
      int nHBF = mSampler.fillHBIRvector(HBIRVec, lastIR, intRecord);
      lastIR = intRecord + 1;

      if (nHBF) {
        for (int j = 0; j < nHBF - 1; j++) {
          o2::InteractionRecord rdhIR = HBIRVec[j];
          for (int link = 0; link < (int)mPages.size(); ++link) {
            setRDH(mPages[link].mRDH, link, rdhIR);
            mPages[link].flush(mFileDest);
          }
        }
      

      uint32_t current_orbit = intRecord.orbit;
      LOG(DEBUG) << "old orbit " << old_orbit << " new orbit " << current_orbit;
      if (old_orbit != current_orbit) {
        for (DataPageWriter& writer : mPages)
          writer.flush(mFileDest);
        for (int nlink = 0; nlink < NPMs; ++nlink)
          setRDH(mPages[nlink].mRDH, nlink, intRecord);
        old_orbit = current_orbit;
      }
      auto channels = bcd.getBunchChannelData(ft0ChData);
      int nch = channels.size();
      if (nch) {

        convertDigits(channels, lut, intRecord);
      }
    }
  }
  for (DataPageWriter& writer : mPages)
    writer.flush(mFileDest);
  for (int nlink = 0; nlink < NPMs; ++nlink)
    setRDH(mPages[nlink].mRDH, nlink, intRecord);
  
  }
}
/*******************************************************************************************************************/
void Digits2Raw::convertDigits(const o2::ft0::ChannelData& channels, const o2::ft0::LookUpTable& lut, o2::InteractionRecord const& intRecord)
{

  // check empty event
  int nch = channels.size();
  for (int ich = 0; ich < nch; ich++) {
    channels[ich].print();
    int oldlink = -1;
    int nchannels = 0;
    for (auto& d : mTimeAmp) {
      int nlink = lut.getLink(d.ChId);
      if (nlink != oldlink) {
        if (oldlink >= 0) {
          uint nGBTWords = uint((nchannels + 1) / 2);
          if ((nchannels % 2) == 1)
            mRawEventData.mEventData[nchannels] = {};
          mRawEventData.mEventHeader.nGBTWords = nGBTWords;
          mPages[oldlink].write(mRawEventData.to_vector(0));
        }
        oldlink = nlink;
        mRawEventData.mEventHeader = makeGBTHeader(nlink, intRecord);
        nchannels = 0;
      }
      auto& newData = mRawEventData.mEventData[nchannels];
      bool isAside = (channels.ChId < 96);
      newData.charge = channels.QTCAmpl;
      newData.time = channels.CFDTime;
      newData.is1TimeLostEvent = 0;
      newData.is2TimeLostEvent = 0;
      newData.isADCinGate = 1;
      newData.isAmpHigh = 0;
      newData.isDoubleEvent = 0;
      newData.isEventInTVDC = digit.getisVrtx() ? 1 : 0;
      newData.isTimeInfoLate = 0;
      newData.isTimeInfoLost = 0;
      int chain = std::rand() % 2;
      newData.numberADC = chain ? 1 : 0;
      newData.channelID = lut.getMCP(d.ChId);
      LOG(DEBUG) << "packed GBT " << nlink << " channelID   " << (int)newData.channelID << " charge " << newData.charge << " time " << newData.time << " chain " << int(newData.numberADC) << " size " << sizeof(newData);
      nchannels++;
    }
    // fill mEventData[nchannels] with 0s to flag that this is a dummy data
    uint nGBTWords = uint((nchannels + 1) / 2);
    if ((nchannels % 2) == 1)
      mRawEventData.mEventData[nchannels] = {};
    mRawEventData.mEventHeader.nGBTWords = nGBTWords;
    mPages[oldlink].write(mRawEventData.to_vector(0));
    LOG(DEBUG) << " last " << oldlink;
       //TCM
     mRawEventData.mEventHeader = makeGBTHeader(LinkTCM, intRecord); //TCM
    mRawEventData.mEventHeader.nGBTWords = 1;
        auto& tcmdata = mRawEventData.mTCMdata;
        tcmdata.vertex = mTriggers.vertex();
        tcmdata.orA = mTriggers.orA();
        tcmdata.orC = mTriggers.orC();
        tcmdata.sCen = mTrigger.sCnt();
        tcmdata.cen = mTriggers.cen();
        tcmdata.nChanA = mTriggers.nChanA();
        tcmdata.nChanC = mTriggers.nChanC();
        tcmdata.amplA = mTriggers.amplA();
        tcmdata.amplC = mTriggers.amplC();
        tcmdata.timeA = mTriggers.timeA();
        tcmdata.timeC = mTriggers.timeC();
     LOG(DEBUG) << "TCMdata"
               << " time A " << int(tcmdata.timeA) << " time C " << int(tcmdata.timeC)
               << " amp A " << int(tcmdata.amplA) << " amp C " << int(tcmdata.amplC)
               << " N A " << int(tcmdata.nChanA) << " N C " << int(tcmdata.nChanC)
               << " trig "
               << " ver " << tcmdata.vertex << " A " << tcmdata.orA << " C " << tcmdata.orC
               << " size " << sizeof(tcmdata);
    mPages.at(LinkTCM).write(mRawEventData.to_vector(1));
    LOG(DEBUG) << " write TCM " << LinkTCM;
  }
}

//_____________________________________________________________________________________
EventHeader makeGBTHeader(int link, o2::InteractionRecord const& mIntRecord)
{
  EventHeader mEventHeader{};
  mEventHeader.startDescriptor = 0xf;
  mEventHeader.reservedField1 = 0;
  mEventHeader.reservedField2 = 0;
  mEventHeader.bc = mIntRecord.bc;
  mEventHeader.orbit = mIntRecord.orbit;
  LOG(DEBUG) << " makeGBTHeader " << link << " orbit " << mEventHeader.orbit << " BC " << mEventHeader.bc;
  return mEventHeader;
}
//_____________________________________________________________________________
void Digits2Raw::setRDH(o2::header::RAWDataHeader& rdh, int nlink, o2::InteractionRecord rdhIR)
{
  rdh = mSampler.createRDH<o2::header::RAWDataHeader>(rdhIR);
  //rdh.triggerOrbit = rdh.heartbeatOrbit = mIntRecord.orbit;
  //rdh.triggerBC = rdh.heartbeatBC = mIntRecord.bc;
  rdh.linkID = nlink;
  rdh.feeId = nlink;

  rdh.triggerType = o2::trigger::PhT; // ??
  rdh.detectorField = 0xffff;         //empty for FIt yet
  rdh.blockLength = 0xffff;           // ITS keeps this dummy
  rdh.stop = 0;                       // ??? last package  on page
}
//_____________________________________________________________________________

void Digits2Raw::close()
{
  if (mFileDest.is_open())
    mFileDest.close();
}
