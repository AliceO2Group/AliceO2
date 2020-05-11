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
#include "DataFormatsFV0/RawEventData.h"
#include "FV0Simulation/Digits2Raw.h"
#include "CommonConstants/Triggers.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "CommonUtils/StringUtils.h"
#include <Framework/Logger.h>
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <vector>
#include <bitset>
#include <iomanip>
#include "TFile.h"
#include "TTree.h"
#include <gsl/span>
#include "FairLogger.h"

using namespace o2::fv0;

ClassImp(Digits2Raw);

Digits2Raw::Digits2Raw(const std::string& fileRaw, const std::string& fileDigitsName)
{
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogScreenLevel("INFO");
  Digits2Raw::readDigits(fileRaw.c_str(), fileDigitsName.c_str());
}

void Digits2Raw::readDigits(const std::string& outDir, const std::string& fileDigitsName)
{
  LOG(INFO) << "==============FV0: Digits2Raw::convertDigits" << std::endl;

  o2::fv0::LookUpTable lut{o2::fv0::Digits2Raw::linear()};
  LOG(DEBUG) << " ##### LookUp set" << std::endl;

  std::string outd = outDir;
  if (outd.back() != '/') {
    outd += '/';
  }
  using namespace o2::raw;
  for (int ilink = 0; ilink < N_PMS; ++ilink) {
    mLinkID = uint32_t(ilink);
    mFeeID = uint64_t(ilink);
    mCruID = uint16_t(0);
    mEndPointID = uint32_t(0);
    std::string outFileLink = mOutputPerLink ? o2::utils::concat_string(outDir, "fv0_link", std::to_string(ilink), ".raw") : o2::utils::concat_string(outDir, "fv0.raw");
    LOG(INFO) << "register link " << outFileLink;
    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outFileLink);
  }

  TFile* fdig = TFile::Open(fileDigitsName.data());
  assert(fdig != nullptr);
  LOG(INFO) << " Open digits file " << fileDigitsName.data();
  TTree* digTree = (TTree*)fdig->Get("o2sim");

  std::vector<o2::fv0::BCData> digitsBC, *fv0BCDataPtr = &digitsBC;
  std::vector<o2::fv0::ChannelData> digitsCh, *fv0ChDataPtr = &digitsCh;

  digTree->SetBranchAddress("FV0DigitBC", &fv0BCDataPtr);
  digTree->SetBranchAddress("FV0DigitCh", &fv0ChDataPtr);

  uint32_t old_orbit = ~0;
  o2::InteractionRecord intRecord;
  o2::InteractionRecord lastIR = mSampler.getFirstIR();
  std::vector<o2::InteractionRecord> HBIRVec;

  for (int ient = 0; ient < digTree->GetEntries(); ient++) {
    digTree->GetEntry(ient);

    int nbc = digitsBC.size();
    LOG(INFO) << "BC size: " << nbc;
    LOG(DEBUG) << "Entry " << ient << " : " << nbc << " BCs stored";
    for (int ibc = 0; ibc < nbc; ibc++) {
      auto& bcd = digitsBC[ibc];
      intRecord = bcd.getIntRecord();
      auto channels = bcd.getBunchChannelData(digitsCh);
      if (!channels.empty())
        convertDigits(bcd, channels, lut, intRecord);
    }
  }
}

void Digits2Raw::convertDigits(o2::fv0::BCData bcdigits, gsl::span<const ChannelData> pmchannels,
                               const o2::fv0::LookUpTable& lut, const o2::InteractionRecord& intRecord)
{
  // check empty event
  int oldlink = -1;
  int nchannels = 0;
  int nch = pmchannels.size();
  LOG(INFO) << "TOTAL CHANNEL: " << nch;
  for (int ich = 0; ich < nch; ich++) {
    //pmchannels[ich].print();
    int nlink = lut.getLink(pmchannels[ich].pmtNumber);
    if (nlink != oldlink) {
      if (oldlink >= 0) {
        uint nGBTWords = uint((nchannels + 1) / 2);
        //LOG(DEBUG) << " oldlink " << oldlink << " nGBTWords " << nGBTWords;
        if ((nchannels % 2) == 1)
          mRawEventData.mEventData[nchannels] = {};
        mRawEventData.mEventHeader.nGBTWords = nGBTWords;
        auto data = mRawEventData.to_vector(false);
        mLinkID = uint32_t(oldlink);
        mFeeID = uint64_t(oldlink);
        LOG(INFO) << "Adding data for link " << oldlink << "\n and"
                  << " Interacrecord" << intRecord;
        mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, intRecord, data);
      }
      oldlink = nlink;
      mRawEventData.mEventHeader = makeGBTHeader(nlink, intRecord);
      nchannels = 0;
      //LOG(INFO) << " switch to new link " << nlink;
    }
    auto& newData = mRawEventData.mEventData[nchannels];
    bool isAside = 1; //(pmchannels[ich].pmtNumber < 48)
    newData.charge = pmchannels[ich].chargeAdc;
    newData.time = pmchannels[ich].time;
    newData.is1TimeLostEvent = 0;
    newData.is2TimeLostEvent = 0;
    newData.isADCinGate = 1;
    newData.isAmpHigh = 0;
    newData.isDoubleEvent = 0;
    newData.isEventInTVDC = 1;
    newData.isTimeInfoLate = 0;
    newData.isTimeInfoLost = 0;
    int chain = std::rand() % 2;
    newData.numberADC = chain ? 1 : 0;
    newData.channelID = lut.getPmChannel(pmchannels[ich].pmtNumber);
    /*LOG(INFO) << "packed GBTlink " << nlink << " channelID   " << (int)newData.channelID << " charge " <<
                      newData.charge << " time " << newData.time << " chain " << int(newData.numberADC) <<
                      " size " << sizeof(newData);*/
    nchannels++;

    if (ich == nch - 1) {
      uint nGBTWords = uint((nchannels + 1) / 2);
      if ((nchannels % 2) == 1)
        mRawEventData.mEventData[nchannels] = {};
      mRawEventData.mEventHeader.nGBTWords = nGBTWords;
      //LOG(INFO)<< "NGBT WORDS  "<<mRawEventData.mEventHeader.nGBTWords;
      auto data = mRawEventData.to_vector(false);
      mLinkID = uint32_t(oldlink);
      mFeeID = uint64_t(oldlink);
      LOG(INFO) << "Adding data for link " << oldlink << "\n and"
                << " Interacrecord" << intRecord;
      mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, intRecord, data);
    }
  }
  // fill mEventData[nchannels] with 0s to flag that this is a dummy data
  uint nGBTWords = uint((nchannels + 1) / 2);
  if ((nchannels % 2) == 1)
    mRawEventData.mEventData[nchannels] = {};
  mRawEventData.mEventHeader.nGBTWords = nGBTWords;
  LOG(DEBUG) << " last link: " << oldlink;
  //TCM
  mRawEventData.mEventHeader = makeGBTHeader(LinkTCM, intRecord); //TCM
  mRawEventData.mEventHeader.nGBTWords = 1;
  auto& tcmdata = mRawEventData.mTCMdata;
  //tcmdata = mTriggers;
  tcmdata.vertex = 1;
  tcmdata.orA = 1;
  tcmdata.orC = 0;
  tcmdata.sCen = 0;
  tcmdata.cen = 0;
  tcmdata.nChanA = 0;
  tcmdata.nChanC = 0;
  tcmdata.amplA = 0;
  tcmdata.amplC = 0;
  tcmdata.timeA = 0;
  tcmdata.timeC = 0;
  //LOG(INFO) << "packed GBT " << nlink << " channelID   " << (int)newData.channelID << " charge " << newData.charge << " time " << newData.time << " chain " << int(newData.numberADC) << " size " << sizeof(newData);

  if (mVerbosity > 0) {
    LOG(INFO) << " triggers read "
      /* << " time A " << mTriggers.timeA << " time C " << mTriggers.timeC
              << " amp A " << ampA << " amp C " << ampC
              << " N A " << int(mTriggers.nChanA) << " N C " << int(mTriggers.nChanC)
              << " trig "
              << " ver " << mTriggers.getVertex() << " A " << mTriggers.getOrA() << " C " << mTriggers.getOrC()*/
      ;

    LOG(INFO) << "TCMdata"
      /*<< " time A " << tcmdata.timeA << " time C " << tcmdata.timeC
              << " amp A " << tcmdata.amplA << " amp C " << tcmdata.amplC
              << " N A " << int(tcmdata.nChanA) << " N C " << int(tcmdata.nChanC)
              << " trig "
              << " ver " << tcmdata.vertex << " A " << tcmdata.orA << " C " << tcmdata.orC
              << " size " << sizeof(tcmdata)*/
      ;
  }

  auto data = mRawEventData.to_vector(kTRUE); //for tcm module
  mLinkID = uint32_t(LinkTCM);
  mFeeID = uint64_t(LinkTCM);
  mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, intRecord, data);
  LOG(INFO) << " write TCM " << LinkTCM;
}

//_____________________________________________________________________________________
EventHeader Digits2Raw::makeGBTHeader(int link, o2::InteractionRecord const& mIntRecord)
{
  EventHeader mEventHeader{};
  mEventHeader.startDescriptor = 0xf;
  mEventHeader.reservedField1 = 0;
  mEventHeader.reservedField2 = 0;
  mEventHeader.bc = mIntRecord.bc;
  mEventHeader.orbit = mIntRecord.orbit;
  if (mVerbosity > 0) {
    LOG(INFO) << " makeGBTHeader " << link << " orbit " << mEventHeader.orbit << " BC " << mEventHeader.bc;
  }
  return mEventHeader;
}
