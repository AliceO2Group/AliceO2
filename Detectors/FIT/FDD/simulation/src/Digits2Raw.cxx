// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//  RAW data format - DataFormat/Detectors/FIT/FDD/RawEventData

#include "FDDBase/Constants.h"
#include "FDDSimulation/Digits2Raw.h"
#include <TTree.h>
#include <cassert>

using namespace o2::fdd;

ClassImp(Digits2Raw);

//_____________________________________________________________________________________
void Digits2Raw::readDigits(const std::string& outDir, const std::string& fileDigitsName)
{
  LOG(INFO) << "==============FDD: Digits2Raw::convertDigits" << std::endl;
  mWriter.setCarryOverCallBack(this);
  LookUpTable lut(true);

  std::string outd = outDir;
  if (outd.back() != '/') {
    outd += '/';
  }

  // Register PM links linearly
  for (int iPmLink = 0; iPmLink < Nmodules; ++iPmLink) {
    uint16_t feeId = uint16_t(iPmLink);
    uint8_t linkId = uint8_t(iPmLink);
    std::string outFileLink = mOutputPerLink ? (outd + "fdd_link" + std::to_string(iPmLink) + ".raw") : (outd + "fdd.raw");
    LOG(INFO) << " Register PM link: " << iPmLink << " to file: " << outFileLink;
    mWriter.registerLink(feeId, sCruId, linkId, sEndPointId, outFileLink);
  }

  // Register TCM link separately
  std::string outFileLink = mOutputPerLink ? (outd + "fdd_link" + std::to_string(sTcmLink) + ".raw") : (outd + "fdd.raw");
  LOG(INFO) << " Register TCM link: " << outFileLink;
  mWriter.registerLink(uint16_t(sTcmLink), sCruId, sTcmLink, sEndPointId, outFileLink);

  TFile* fdig = TFile::Open(fileDigitsName.data());
  assert(fdig != nullptr);
  LOG(INFO) << "Open digits file: " << fileDigitsName.data();

  TTree* digTree = (TTree*)fdig->Get("o2sim");
  std::vector<o2::fdd::Digit> digitsBC, *fddDigitPtr = &digitsBC;
  std::vector<o2::fdd::ChannelData> digitsCh, *fddChDataPtr = &digitsCh;
  digTree->SetBranchAddress("FDDDigit", &fddDigitPtr);
  digTree->SetBranchAddress("FDDDigitCh", &fddChDataPtr);

  for (int ient = 0; ient < digTree->GetEntries(); ient++) {
    digTree->GetEntry(ient);
    int nbc = digitsBC.size();
    for (int ibc = 0; ibc < nbc; ibc++) {
      auto& bcd = digitsBC[ibc];
      auto channels = bcd.getBunchChannelData(digitsCh);

      if (!channels.empty()) {
        LOG(DEBUG) << "o2::fdd::Digits2Raw::readDigits(): Start to convertDigits() at ibc = " << ibc << "  " << bcd.mIntRecord
                   << " iCh0:" << bcd.ref.getFirstEntry() << "  nentries:" << bcd.ref.getEntries();
        convertDigits(bcd, channels, lut);
      }
    }
  }
}
//_____________________________________________________________________________________
void Digits2Raw::convertDigits(o2::fdd::Digit bcdigits, gsl::span<const ChannelData> pmchannels,
                               const o2::fdd::LookUpTable& lut)
{
  const o2::InteractionRecord intRecord = bcdigits.getIntRecord();
  int prevPmLink = -1;
  int iChannelPerLink = 0;
  int nch = pmchannels.size();

  std::stringstream ss;
  ss << "  Number of channels: " << nch << "   (Ch, PMT, Q, T)\n";
  for (int ich = 0; ich < nch; ich++) {
    if (pmchannels[ich].mChargeADC != 0) {
      ss << "          " << std::setw(2) << ich
         << std::setw(3) << pmchannels[ich].mPMNumber
         << std::setw(5) << pmchannels[ich].mChargeADC
         << std::setw(7) << std::setprecision(3) << pmchannels[ich].mTime << "\n";
    }
  }
  LOG(DEBUG) << ss.str().substr(0, ss.str().size() - 1);

  for (int ich = 0; ich < nch; ich++) {
    int nLinkPm = lut.getLink(pmchannels[ich].mPMNumber);
    if (nLinkPm != prevPmLink) {
      if (prevPmLink >= 0) {
        fillSecondHalfWordAndAddData(iChannelPerLink, prevPmLink, intRecord);
      }
      makeGBTHeader(mRawEventData.mEventHeader, nLinkPm, intRecord);
      iChannelPerLink = 0;
      prevPmLink = nLinkPm;
    }
    LOG(DEBUG) << "    Store data for channel: " << ich << " PmLink = " << nLinkPm << "  ";
    auto& newData = mRawEventData.mEventData[iChannelPerLink];
    newData.charge = pmchannels[ich].mChargeADC;
    newData.time = pmchannels[ich].mTime;

    newData.numberADC = bool(pmchannels[ich].mFEEBits & ChannelData::kNumberADC);
    newData.isDoubleEvent = bool(pmchannels[ich].mFEEBits & ChannelData::kIsDoubleEvent);
    newData.isTimeInfoNOTvalid = bool(pmchannels[ich].mFEEBits & ChannelData::kIsTimeInfoNOTvalid);
    newData.isCFDinADCgate = bool(pmchannels[ich].mFEEBits & ChannelData::kIsCFDinADCgate);
    newData.isTimeInfoLate = bool(pmchannels[ich].mFEEBits & ChannelData::kIsTimeInfoLate);
    newData.isAmpHigh = bool(pmchannels[ich].mFEEBits & ChannelData::kIsAmpHigh);
    newData.isEventInTVDC = bool(pmchannels[ich].mFEEBits & ChannelData::kIsEventInTVDC);
    newData.isTimeInfoLost = bool(pmchannels[ich].mFEEBits & ChannelData::kIsTimeInfoLost);

    newData.channelID = lut.getModChannel(pmchannels[ich].mPMNumber);
    iChannelPerLink++;
    if (ich == nch - 1) {
      fillSecondHalfWordAndAddData(iChannelPerLink, prevPmLink, intRecord);
    }
  }

  // TCM
  makeGBTHeader(mRawEventData.mEventHeader, sTcmLink, intRecord);
  mRawEventData.mEventHeader.nGBTWords = 1;
  auto& tcmdata = mRawEventData.mTCMdata;
  mTriggers = bcdigits.mTriggers;

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

  auto data = mRawEventData.to_vector(kTRUE); //for tcm module
  uint32_t linkId = uint32_t(sTcmLink);
  uint64_t feeId = uint64_t(sTcmLink);
  mWriter.addData(feeId, sCruId, linkId, sEndPointId, intRecord, data);

  // fill mEventData[iChannelPerLink] with 0s to flag that this is a dummy data
  uint nGBTWords = uint((iChannelPerLink + 1) / 2);
  if ((iChannelPerLink % 2) == 1) {
    mRawEventData.mEventData[iChannelPerLink] = {};
  }
  mRawEventData.mEventHeader.nGBTWords = nGBTWords;
  //  LOG(DEBUG) << " last link: " << prevPmLink;
}
//_____________________________________________________________________________________
void Digits2Raw::makeGBTHeader(EventHeader& eventHeader, int link, o2::InteractionRecord const& mIntRecord)
{
  eventHeader.startDescriptor = 0xf;
  eventHeader.reservedField1 = 0;
  eventHeader.reservedField2 = 0;
  eventHeader.reservedField3 = 0;
  eventHeader.bc = mIntRecord.bc;
  eventHeader.orbit = mIntRecord.orbit;
  LOG(DEBUG) << "  makeGBTHeader for link: " << link;
}
//_____________________________________________________________________________________
void Digits2Raw::fillSecondHalfWordAndAddData(int iChannelPerLink, int prevPmLink, const o2::InteractionRecord& ir)
{
  uint nGBTWords = uint((iChannelPerLink + 1) / 2);
  if ((iChannelPerLink % 2) == 1) {
    mRawEventData.mEventData[iChannelPerLink] = {};
    LOG(DEBUG) << "    Fill up empty second half-word.";
  }
  mRawEventData.mEventHeader.nGBTWords = nGBTWords;
  auto data = mRawEventData.to_vector(false);
  uint32_t linkId = uint32_t(prevPmLink);
  uint64_t feeId = uint64_t(prevPmLink);
  mWriter.addData(feeId, sCruId, linkId, sEndPointId, ir, data);
  LOG(DEBUG) << "  Switch prevPmLink:  " << prevPmLink << ". Save data with nGBTWords="
             << nGBTWords << " in header. Last channel: " << iChannelPerLink;
}

//_____________________________________________________________________________________
int Digits2Raw::carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                                const char* ptr, int maxSize, int splitID,
                                std::vector<char>& trailer, std::vector<char>& header) const
{
  return 0; // do not split, always start new CRU page
}
