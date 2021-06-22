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
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "CommonUtils/StringUtils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <Framework/Logger.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cassert>
#include <fstream>
#include <vector>
#include <bitset>
#include <iomanip>
#include "TFile.h"
#include "TTree.h"
#include <gsl/span>

using namespace o2::ft0;
using CcdbManager = o2::ccdb::BasicCCDBManager;

ClassImp(Digits2Raw);

EventHeader makeGBTHeader(int link, o2::InteractionRecord const& mIntRecord);

Digits2Raw::Digits2Raw(const std::string& fileRaw, const std::string& fileDigitsName)
{
  readDigits(fileRaw.c_str(), fileDigitsName.c_str());
}

void Digits2Raw::readDigits(const std::string& outDir, const std::string& fileDigitsName)
{
  LOG(INFO) << "**********Digits2Raw::convertDigits" << std::endl;
  mWriter.setCarryOverCallBack(this);

  o2::ft0::LookUpTable lut{o2::ft0::LookUpTable::readTable()};
  mLinkTCM = lut.getLink(lut.getTCMchannel());
  LOG(INFO) << " ##### LookUp set, TCM " << mLinkTCM;
  std::string outd = outDir;
  if (outd.back() != '/') {
    outd += '/';
  }
  using namespace o2::raw;
  for (int ilink = 0; ilink < NPMs - 1; ++ilink) {
    if (ilink < 8) {
      mLinkID = uint32_t(ilink);
      mCruID = uint16_t(0);
      mEndPointID = uint32_t(0);
      mFeeID = uint64_t(ilink);
    } else {
      mLinkID = uint32_t(ilink) - 8;
      mCruID = uint16_t(0);
      mEndPointID = uint32_t(1);
      mFeeID = uint64_t(ilink);
    }
    std::string outFileLink = mOutputPerLink ? o2::utils::Str::concat_string(outDir, "ft0_link", std::to_string(ilink), ".raw") : o2::utils::Str::concat_string(outDir, "ft0.raw");
    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outFileLink);
    LOG(INFO) << " registered links " << mLinkID << " endpoint " << mEndPointID;
  }
  //TCM
  std::string outFileLink = mOutputPerLink ? o2::utils::Str::concat_string(outDir, "ft0_link", std::to_string(NPMs - 1), ".raw") : o2::utils::Str::concat_string(outDir, "ft0.raw");
  mWriter.registerLink(mLinkTCM + 8, mCruID, mLinkTCM, 0, outFileLink);
  LOG(INFO) << " registered link  TCM " << mLinkTCM;

  TFile* fdig = TFile::Open(fileDigitsName.data());
  assert(fdig != nullptr);
  LOG(INFO) << " Open digits file " << fileDigitsName.data();
  TTree* digTree = (TTree*)fdig->Get("o2sim");

  std::vector<o2::ft0::Digit> digitsBC, *ft0BCDataPtr = &digitsBC;
  std::vector<o2::ft0::ChannelData> digitsCh, *ft0ChDataPtr = &digitsCh;

  digTree->SetBranchAddress("FT0DIGITSBC", &ft0BCDataPtr);
  digTree->SetBranchAddress("FT0DIGITSCH", &ft0ChDataPtr);

  uint32_t old_orbit = ~0;
  o2::InteractionRecord intRecord;
  o2::InteractionRecord lastIR = mSampler.getFirstIR();
  std::vector<o2::InteractionRecord> HBIRVec;

  for (int ient = 0; ient < digTree->GetEntries(); ient++) {
    digTree->GetEntry(ient);

    int nbc = digitsBC.size();
    LOG(INFO) << "Entry " << ient << " : " << nbc << " BCs stored";
    for (int ibc = 0; ibc < nbc; ibc++) {
      auto& bcd = digitsBC[ibc];
      intRecord = bcd.getIntRecord();
      auto channels = bcd.getBunchChannelData(digitsCh);
      if (!channels.empty()) {
        convertDigits(bcd, channels, lut, intRecord);
      }
    }
  }
}

void Digits2Raw::convertDigits(o2::ft0::Digit bcdigits,
                               gsl::span<const ChannelData> pmchannels,
                               const o2::ft0::LookUpTable& lut,
                               o2::InteractionRecord const& intRecord)
{

  // check empty event
  int oldlink = -1;
  int oldendpoint = -1;
  int nchannels = 0;
  int nch = pmchannels.size();
  for (int ich = 0; ich < nch; ich++) {
    int nlink = lut.getLink(pmchannels[ich].ChId);
    int ep = lut.getEP(pmchannels[ich].ChId);
    if (nlink != oldlink || ep != oldendpoint) {
      if (oldlink >= 0) {
        uint nGBTWords = uint((nchannels + 1) / 2);
        LOG(DEBUG) << " oldlink " << oldlink << " old EP " << oldendpoint << " nGBTWords " << nGBTWords << " new link " << nlink << " ep  " << ep;
        if ((nchannels % 2) == 1) {
          mRawEventData.mEventData[nchannels] = {};
        }
        mRawEventData.mEventHeader.nGBTWords = nGBTWords;
        auto data = mRawEventData.to_vector(false);
        mLinkID = uint32_t(oldlink);
        mFeeID = uint64_t(oldlink);
        mEndPointID = uint32_t(oldendpoint);
        if (mEndPointID == 1) {
          mFeeID += 8;
        }
        LOG(DEBUG) << " new link start " << mFeeID << " " << mCruID << " " << mLinkID << " " << mEndPointID;
        mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, intRecord, data);
      }
      oldlink = nlink;
      oldendpoint = ep;
      mRawEventData.mEventHeader = makeGBTHeader(nlink, intRecord);
      nchannels = 0;
      LOG(DEBUG) << " switch to new link " << nlink << " EP " << ep;
    }
    auto& newData = mRawEventData.mEventData[nchannels];
    bool isAside = (pmchannels[ich].ChId < 96);
    newData.charge = pmchannels[ich].QTCAmpl;
    newData.time = pmchannels[ich].CFDTime;
    newData.generateFlags();
    newData.channelID = lut.getMCP(pmchannels[ich].ChId);
    LOG(DEBUG) << " ID " << int(pmchannels[ich].ChId) << " packed GBT " << nlink << " channelID   " << (int)newData.channelID << " charge " << newData.charge << " time " << newData.time << " chain " << int(newData.numberADC)
               << " size " << sizeof(newData) << " mEndPointID " << ep;
    nchannels++;
  }
  // fill mEventData[nchannels] with 0s to flag that this is a dummy data
  uint nGBTWords = uint((nchannels + 1) / 2);
  if ((nchannels % 2) == 1) {
    mRawEventData.mEventData[nchannels] = {};
  }
  mRawEventData.mEventHeader.nGBTWords = nGBTWords;
  auto datalast = mRawEventData.to_vector(false);
  mLinkID = uint32_t(oldlink);
  mFeeID = uint64_t(oldlink);
  mEndPointID = uint32_t(oldendpoint);
  if (mEndPointID == 1) {
    mFeeID += 8;
  }
  mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, intRecord, datalast);
  LOG(DEBUG) << " last " << mFeeID << " " << mCruID << " " << mLinkID << " " << mEndPointID;
  //TCM
  mRawEventData.mEventHeader = makeGBTHeader(mLinkTCM, intRecord); //TCM
  mRawEventData.mEventHeader.nGBTWords = 1;
  auto& tcmdata = mRawEventData.mTCMdata;
  mTriggers = bcdigits.getTriggers();

  float ampA = mTriggers.amplA;
  float ampC = mTriggers.amplC;
  if (ampA > 131071) {
    ampA = 131071; //2^17
  }
  if (ampC > 131071) {
    ampC = 131071; //2^17
  }
  tcmdata.vertex = mTriggers.getVertex();
  tcmdata.orA = mTriggers.getOrA();
  tcmdata.orC = mTriggers.getOrC();
  tcmdata.sCen = mTriggers.getSCen();
  tcmdata.cen = mTriggers.getCen();
  tcmdata.nChanA = mTriggers.nChanA;
  tcmdata.nChanC = mTriggers.nChanC;
  tcmdata.amplA = ampA;
  tcmdata.amplC = ampC;
  tcmdata.timeA = mTriggers.timeA;
  tcmdata.timeC = mTriggers.timeC;
  LOG(DEBUG) << " TCM  triggers read "
             << " time A " << mTriggers.timeA << " time C " << mTriggers.timeC
             << " amp A " << ampA << " amp C " << ampC
             << " N A " << int(mTriggers.nChanA) << " N C " << int(mTriggers.nChanC)
             << " trig "
             << " ver " << mTriggers.getVertex() << " A " << mTriggers.getOrA() << " C " << mTriggers.getOrC();

  LOG(DEBUG) << "TCMdata"
             << " time A " << tcmdata.timeA << " time C " << tcmdata.timeC
             << " amp A " << tcmdata.amplA << " amp C " << tcmdata.amplC
             << " N A " << int(tcmdata.nChanA) << " N C " << int(tcmdata.nChanC)
             << " trig "
             << " ver " << tcmdata.vertex << " A " << tcmdata.orA << " C " << tcmdata.orC
             << " size " << sizeof(tcmdata);

  auto data = mRawEventData.to_vector(1);
  mLinkID = uint32_t(mLinkTCM);
  mFeeID = uint64_t(mLinkTCM) + 8;
  mEndPointID = 0;
  mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, intRecord, data);
  LOG(DEBUG) << " TCM " << mFeeID << " " << mCruID << " " << mLinkID << " " << mEndPointID;
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

//_____________________________________________________________________________________
int Digits2Raw::carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                                const char* ptr, int maxSize, int splitID,
                                std::vector<char>& trailer, std::vector<char>& header) const
{
  return 0; // do not split, always start new CRU page
}
