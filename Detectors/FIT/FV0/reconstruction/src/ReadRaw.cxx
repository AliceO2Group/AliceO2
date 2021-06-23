// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FV0Reconstruction/ReadRaw.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/Logger.h"
#include "TFile.h"

using namespace o2::fv0;
using RDHUtils = o2::raw::RDHUtils;

ClassImp(ReadRaw);

ReadRaw::ReadRaw(bool doConversionToDigits, const std::string inputRawFilePath, const std::string outputDigitsFilePath)
{
  LOG(INFO) << "o2::fv0::ReadRaw::ReadRaw(): Read Raw file: " << inputRawFilePath.data() << " and convert to: " << outputDigitsFilePath.data();
  mRawFileIn.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  mRawFileIn.open(inputRawFilePath, std::fstream::in | std::fstream::binary);
  LookUpTable lut(true);
  ReadRaw::readRawData(lut);
  ReadRaw::writeDigits(outputDigitsFilePath.data());
}

void ReadRaw::readRawData(const LookUpTable& lut)
{
  LOG(INFO) << "o2::fv0::ReadRaw::readRawData():Start.";
  constexpr int CRUWordSize = 16;
  o2::header::RAWDataHeader mRDH;
  ChannelData chData;                             // output to store digits
  EventHeader eventHeader;                        // raw data container
  TCMdata mTCMdata;                               // raw data container
  EventData eventData[Constants::nChannelsPerPm]; // raw data container

  // Get input RAW file size
  mRawFileIn.seekg(0, mRawFileIn.end);
  long rawFileSize = mRawFileIn.tellg();
  mRawFileIn.seekg(0);

  long posInFile = 0;
  while (posInFile < rawFileSize - sizeof(mRDH)) {
    // Read contents of the next RDH
    mRawFileIn.seekg(posInFile);
    mRawFileIn.read(reinterpret_cast<char*>(&mRDH), sizeof(mRDH));
    int nwords = RDHUtils::getMemorySize(mRDH);
    int offset = RDHUtils::getOffsetToNext(mRDH);
    int link = RDHUtils::getLinkID(mRDH);
    posInFile += offset; // posInFile is now prepared for the next iteration of the main loop

    int posPayload = 0;
    if (nwords > sizeof(mRDH)) {
      // RDHUtils::printRDH(mRDH);
      // Read the payload following the RDH (only if there is anything to read)
      posPayload = int(sizeof(mRDH));

      while (posPayload < nwords) {
        mRawFileIn.read(reinterpret_cast<char*>(&eventHeader), sizeof(eventHeader));
        posPayload += sizeof(eventHeader);
        LOG(DEBUG) << "  Read internal EventHeader for link: " << link
                   << "  nWords: " << (int)eventHeader.nGBTWords
                   << "  orbit: " << int(eventHeader.orbit)
                   << "  BC: " << int(eventHeader.bc)
                   << "  posInFile: " << posInFile
                   << "  posPayload: " << posPayload;
        o2::InteractionRecord intrec{uint16_t(eventHeader.bc), uint32_t(eventHeader.orbit)};

        if (link == lut.getTcmLink()) { // is TCM payload
          mRawFileIn.read(reinterpret_cast<char*>(&mTCMdata), sizeof(mTCMdata));
          posPayload += sizeof(mTCMdata);
          LOG(DEBUG) << "    Read TCM: posPayload: " << posPayload
                     << " posInFile: " << posInFile;
        } else {                                                           // is PM payload
          posPayload += CRUWordSize - o2::fv0::RawEventData::sPayloadSize; // padding is enabled
          for (int i = 0; i < eventHeader.nGBTWords; ++i) {
            mRawFileIn.read(reinterpret_cast<char*>(&eventData[2 * i]), o2::fv0::RawEventData::sPayloadSizeFirstWord);
            posPayload += o2::fv0::RawEventData::sPayloadSizeFirstWord;
            chData = {Short_t(lut.getChannel(link, int(eventData[2 * i].channelID))),
                      Float_t(eventData[2 * i].time),
                      Short_t(eventData[2 * i].charge)};
            mDigitAccum[intrec].emplace_back(chData);
            LOG(DEBUG) << "    Read 1st half-word: (PMchannel, globalChannel, Q, T, posPayload) =  "
                       << std::setw(3) << int(eventData[2 * i].channelID)
                       << std::setw(4) << lut.getChannel(link, int(eventData[2 * i].channelID))
                       << std::setw(5) << int(eventData[2 * i].charge)
                       << std::setw(5) << float(eventData[2 * i].time)
                       << std::setw(5) << posPayload;

            Short_t channelIdFirstHalfWord = chData.pmtNumber;

            mRawFileIn.read(reinterpret_cast<char*>(&eventData[2 * i + 1]), o2::fv0::RawEventData::sPayloadSizeSecondWord);
            posPayload += o2::fv0::RawEventData::sPayloadSizeSecondWord;
            chData = {Short_t(lut.getChannel(link, (eventData[2 * i + 1].channelID))),
                      Float_t(eventData[2 * i + 1].time),
                      Short_t(eventData[2 * i + 1].charge)};
            if (chData.pmtNumber <= channelIdFirstHalfWord) {
              // Don't save the second half-word if it is only filled with zeroes (empty-data)
              // TODO: Verify if it works correctly with real data from readout
              continue;
            }
            mDigitAccum[intrec].emplace_back(chData);
            LOG(DEBUG) << "    Read 2nd half-word: (PMchannel, globalChannel, Q, T, posPayload) =  "
                       << std::setw(3) << int(eventData[2 * i + 1].channelID)
                       << std::setw(4) << lut.getChannel(link, int(eventData[2 * i + 1].channelID))
                       << std::setw(5) << int(eventData[2 * i + 1].charge)
                       << std::setw(5) << float(eventData[2 * i + 1].time)
                       << std::setw(5) << posPayload;
          }
        }
      }
    }
  }
  close();
  LOG(INFO) << "o2::fv0::ReadRaw::readRawData():Finished.";
}

void ReadRaw::close()
{
  if (mRawFileIn.is_open()) {
    mRawFileIn.close();
  }
}

void ReadRaw::writeDigits(const std::string& outputDigitsFilePath)
{
  TFile* outFile = new TFile(outputDigitsFilePath.data(), "RECREATE");
  if (!outFile || outFile->IsZombie()) {
    LOG(ERROR) << "Failed to open " << outputDigitsFilePath << " output file";
  } else {
    LOG(INFO) << "o2::fv0::ReadRaw::writeDigits(): Opened output file: " << outputDigitsFilePath;
  }
  TTree* outTree = new TTree("o2sim", "o2sim");
  std::vector<ChannelData> chDataVecTree;
  std::vector<BCData> chBcVecTree;

  for (auto& digit : mDigitAccum) {
    LOG(DEBUG) << " IR (" << digit.first << ")   (i, PMT, Q, T):";
    for (uint16_t i = 0; i < digit.second.size(); i++) {
      ChannelData* chd = &(digit.second.at(i));
      LOG(DEBUG) << "  " << std::setw(3) << i
                 << std::setw(4) << chd->pmtNumber
                 << std::setw(5) << chd->chargeAdc
                 << std::setw(5) << chd->time;
    }

    size_t nStored = 0;
    size_t first = chDataVecTree.size();
    for (auto& sec : digit.second) {
      chDataVecTree.emplace_back(int(sec.pmtNumber), float(sec.time), Short_t(sec.chargeAdc));
      nStored++;
    }
    Triggers triggers; // TODO: Actual values are not set
    chBcVecTree.emplace_back(first, nStored, digit.first, triggers);
  }

  outTree->Branch("FV0DigitBC", &chBcVecTree);
  outTree->Branch("FV0DigitCh", &chDataVecTree);
  outTree->Fill();

  outFile->cd();
  outTree->Write();
  outFile->Close();
  LOG(INFO) << "o2::fv0::ReadRaw::writeDigits(): Finished converting " << chBcVecTree.size() << " events.";
}
