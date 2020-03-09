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
  uint reservedField : 28;
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
Event header + event data, 2 channels per 1 GBT word;
if no data for this PM - only headers.

Trigger mode : detector sends data to FLP at each trigger;
Continueous mode  :   for only bunches with data at least in 1 channel.  
*/

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/RawEventData.h"
#include "FT0Reconstruction/ReadRaw.h"
#include "CommonConstants/Triggers.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/DigitsTemp.h"
#include "DataFormatsFT0/ChannelData.h"
#include <Framework/Logger.h>
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <bitset>
#include <iomanip>
#include <numeric>
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "CommonConstants/LHCConstants.h"

using namespace o2::ft0;

ClassImp(ReadRaw);

ReadRaw::ReadRaw(const std::string fileRaw, std::string fileDataOut)
{

  LOG(INFO) << " constructor ReadRaw "
            << "file to read " << fileRaw.data() << " file to write " << fileDataOut.data();
  mFileDest.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  mFileDest.open(fileRaw, std::fstream::in | std::fstream::binary);
  o2::ft0::LookUpTable lut{o2::ft0::ReadRaw::linear()};
  ReadRaw::readData(fileRaw.c_str(), lut);
  ReadRaw::writeDigits(fileDataOut.data());
}

void ReadRaw::readData(const std::string fileRaw, const o2::ft0::LookUpTable& lut)
{

  LOG(INFO) << " readData ";
  o2::header::RAWDataHeader mRDH;
  const char padding[CRUWordSize] = {0};
  int nStored = 0;

  std::vector<o2::ft0::DigitsTemp> digits;
  std::vector<o2::ft0::ChannelData>* chDgDataArr = nullptr;
  o2::ft0::ChannelData chData;
  // o2::ft0::Triggers mTriggers;
  uint64_t bctrigger = 0;
  mFileDest.seekg(0, mFileDest.end);
  long sizeFile = mFileDest.tellg();
  mFileDest.seekg(0);
  LOG(DEBUG) << "SizeFile " << sizeFile;

  // read content of infile
  long posInFile = 0;
  while (posInFile < sizeFile - sizeof(mRDH)) {

    int pos = 0;
    mFileDest.seekg(posInFile);
    mFileDest.read(reinterpret_cast<char*>(&mRDH), sizeof(mRDH));
    printRDH(&mRDH);
    int nwords = mRDH.memorySize;
    int npackages = mRDH.packetCounter;
    int numPage = mRDH.pageCnt;
    int offset = mRDH.offsetToNext;
    int link = mRDH.linkID;
    if (nwords <= sizeof(mRDH)) {
      posInFile += mRDH.offsetToNext;
      LOG(INFO) << " next RDH";
      pos = 0;
    } else {
      posInFile += offset;
      pos = int(sizeof(mRDH));

      while (pos < nwords) {
        mFileDest.read(reinterpret_cast<char*>(&mEventHeader), sizeof(mEventHeader));
        pos += sizeof(mEventHeader);
        LOG(DEBUG) << "read  header for " << link << "word " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " BC " << int(mEventHeader.bc) << " pos " << pos << " posinfile " << posInFile;
        o2::InteractionRecord intrec{uint16_t(mEventHeader.bc), uint32_t(mEventHeader.orbit)};
        auto [digitIter, isNew] = mDigitAccum.try_emplace(intrec);
        if (isNew) {
          double eventTime = intrec.bc2ns();
          LOG(DEBUG) << "new intrec " << intrec.orbit << " " << intrec.bc << " link " << link;
          o2::ft0::DigitsTemp& digit = digitIter->second;
          digit.setTime(eventTime);
          digit.setInteractionRecord(intrec);
        }
        chDgDataArr = &digitIter->second.getChDgData(); //&mDigitsTemp.getChDgData();
        if (link == 18) {
          mFileDest.read(reinterpret_cast<char*>(&mTCMdata), sizeof(mTCMdata));
          pos += sizeof(mTCMdata);
          digitIter->second.setTriggers(mTCMdata.orA, mTCMdata.orC, mTCMdata.vertex, mTCMdata.sCen, mTCMdata.cen, mTCMdata.nChanA, mTCMdata.nChanC, mTCMdata.amplA, mTCMdata.amplC, mTCMdata.timeA, mTCMdata.timeC);

          LOG(DEBUG) << "read TCM  " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " BC " << int(mEventHeader.bc) << " pos " << pos << " posinfile " << posInFile;
        } else {
          if (mIsPadded) {
            pos += CRUWordSize - o2::ft0::EventHeader::PayloadSize;
          }
          for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
            mFileDest.read(reinterpret_cast<char*>(&mEventData[2 * i]), o2::ft0::EventData::PayloadSizeFirstWord);
            chDgDataArr->emplace_back(lut.getChannel(link, int(mEventData[2 * i].channelID)),
                                      int(mEventData[2 * i].time),
                                      int(mEventData[2 * i].charge),
                                      int(mEventData[2 * i].numberADC));

            pos += o2::ft0::EventData::PayloadSizeFirstWord;
            LOG(DEBUG) << " read 1st word channelID " << int(mEventData[2 * i].channelID) << " charge " << mEventData[2 * i].charge << " time " << mEventData[2 * i].time << " PM " << link << " lut channel " << lut.getChannel(link, int(mEventData[2 * i].channelID)) << " pos " << pos;

            mFileDest.read(reinterpret_cast<char*>(&mEventData[2 * i + 1]), EventData::PayloadSizeSecondWord);
            pos += o2::ft0::EventData::PayloadSizeSecondWord;
            LOG(DEBUG) << "read 2nd word channel " << int(mEventData[2 * i + 1].channelID) << " charge " << int(mEventData[2 * i + 1].charge) << " time " << mEventData[2 * i + 1].time << " PM " << link << " lut channel " << lut.getChannel(link, int(mEventData[2 * i].channelID)) << " pos " << pos;
            if (mEventData[2 * i + 1].charge <= 0 && mEventData[2 * i + 1].channelID <= 0 && mEventData[2 * i + 1].time <= 0) {
              continue;
            }
            chDgDataArr->emplace_back(lut.getChannel(link, int(mEventData[2 * i + 1].channelID)),
                                      int(mEventData[2 * i + 1].time),
                                      int(mEventData[2 * i + 1].charge),
                                      int(mEventData[2 * i + 1].numberADC));
          }
        }
      }
    }
  }
  close();
}

//_____________________________________________________________________________

void ReadRaw::close()
{
  LOG(INFO) << " CLOSE ";
  if (mFileDest.is_open())
    mFileDest.close();
}
//_____________________________________________________________________________
void ReadRaw::writeDigits(std::string fileDataOut)
{
  TFile* mOutFile = new TFile(fileDataOut.data(), "RECREATE");
  if (!mOutFile || mOutFile->IsZombie()) {
    LOG(ERROR) << "Failed to open " << fileDataOut << " output file";
  } else {
    LOG(INFO) << "Opened " << fileDataOut << " output file";
  }
  TTree* mOutTree = new TTree("o2sim", "o2sim");
  // retrieve the digits from the input
  auto inDigits = mDigitAccum;
  LOG(INFO) << "RECEIVED DIGITS SIZE " << inDigits.size();

  // connect this to a particular branch

  std::vector<o2::ft0::Digit> digitVec;
  std::vector<o2::ft0::ChannelData> chDataVec;
  digitVec.reserve(mDigitAccum.size());
  size_t numberOfChData = 0;
  for (auto const& [intrec, digit] : mDigitAccum)
    numberOfChData += digit.getChDgData().size();
  chDataVec.reserve(numberOfChData);
  for (auto& [intrec, digit] : mDigitAccum) {
    int first = gsl::narrow_cast<int>(chDataVec.size());
    auto& chDgData = digit.getChDgData();
    chDataVec.insert(chDataVec.end(), chDgData.begin(), chDgData.end());
    o2::ft0::Digit newDigit{first, (int)chDgData.size(), intrec, mTrigger};
    digitVec.emplace_back(newDigit);
  }
  mDigitAccum.clear();
  mOutTree->Branch("FT0Digit", &digitVec);
  mOutTree->Branch("FT0ChannelData", &chDataVec);
  mOutTree->Fill();

  mOutFile->cd();
  mOutTree->Write();
  mOutFile->Close();
}
