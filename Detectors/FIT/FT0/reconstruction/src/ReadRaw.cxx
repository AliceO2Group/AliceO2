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
#include "DataFormatsFT0/Digit.h"
#include "FT0Reconstruction/ReadRaw.h"
#include "CommonConstants/Triggers.h"
#include <Framework/Logger.h>
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <bitset>
#include <iomanip>
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
  std::vector<o2::ft0::ChannelData>* chDgDataArr = nullptr;
  o2::ft0::ChannelData chData;
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
    pos += int(sizeof(mRDH));
    posInFile += nwords;

    while (pos < mRDH.memorySize) {
      mFileDest.read(reinterpret_cast<char*>(&mEventHeader), sizeof(mEventHeader));
      pos += sizeof(mEventHeader);
      LOG(DEBUG) << "read  header for " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " BC " << int(mEventHeader.bc) << " pos " << pos << " posinfile " << posInFile;
      o2::InteractionRecord intrec{uint16_t(mEventHeader.bc), uint32_t(mEventHeader.orbit)};
      auto [digitIter, isNew] = mDigitAccum.try_emplace(intrec);
      auto& digits = digitIter->second;
      if (isNew) {
        double eventTime = intrec.bc2ns();
        digits.setTime(eventTime);
        digits.setInteractionRecord(intrec);
        digits.setTriggers(0, 0, 0, 0, 0);
      }
      chDgDataArr = &digits.getChDgData();
      if (mIsPadded) {
        pos += CRUWordSize - o2::ft0::EventHeader::PayloadSize;
      }

      for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
        mFileDest.read(reinterpret_cast<char*>(&mEventData[2 * i]), o2::ft0::EventData::PayloadSizeFirstWord);
        chData.ChId = lut.getChannel(link, int(mEventData[2 * i].channelID));
        chData.CFDTime = mEventData[2 * i].time;
        chData.QTCAmpl = mEventData[2 * i].charge;
        chData.numberOfParticles = mEventData[2 * i].numberADC;
        chDgDataArr->emplace_back(chData);
        pos += o2::ft0::EventData::PayloadSizeFirstWord;
        LOG(DEBUG) << " read 1st word channelID " << int(mEventData[2 * i].channelID) << " charge " << mEventData[2 * i].charge << " time " << mEventData[2 * i].time << " PM " << link << " lut channel " << lut.getChannel(link, int(mEventData[2 * i].channelID)) << " pos " << pos;

        mFileDest.read(reinterpret_cast<char*>(&mEventData[2 * i + 1]), EventData::PayloadSizeSecondWord);
        pos += o2::ft0::EventData::PayloadSizeSecondWord;
        LOG(DEBUG) << "read 2nd word channel " << int(mEventData[2 * i + 1].channelID) << " charge " << int(mEventData[2 * i + 1].charge) << " time " << mEventData[2 * i + 1].time << " PM " << link << " lut channel " << lut.getChannel(link, int(mEventData[2 * i].channelID)) << " pos " << pos;
        if (mEventData[2 * i + 1].charge <= 0 && mEventData[2 * i + 1].channelID <= 0 && mEventData[2 * i + 1].time <= 0) {
          continue;
        }
        chData.ChId = lut.getChannel(link, int(mEventData[2 * i + 1].channelID));
        chData.CFDTime = mEventData[2 * i + 1].time;
        chData.QTCAmpl = mEventData[2 * i + 1].charge;
        chData.numberOfParticles = mEventData[2 * i + 1].numberADC;
        chDgDataArr->emplace_back(chData);
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
  digitVec.reserve(mDigitAccum.size());
  for (auto& [intrec, digit] : mDigitAccum)
    digitVec.emplace_back(std::move(digit));
  mDigitAccum.clear();
  mOutTree->Branch("FT0Digit", &digitVec);
  mOutTree->Fill();

  mOutFile->cd();
  mOutTree->Write();
  mOutFile->Close();
}
