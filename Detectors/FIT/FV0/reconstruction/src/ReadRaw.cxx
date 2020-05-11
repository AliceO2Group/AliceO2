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
#include "DataFormatsFV0/RawEventData.h"
#include "FV0Reconstruction/ReadRaw.h"
#include "CommonConstants/Triggers.h"
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
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

using namespace o2::fv0;

ClassImp(ReadRaw);

ReadRaw::ReadRaw(const std::string fileRaw, std::string fileDataOut)
{

  LOG(INFO) << " constructor ReadRaw: \n"
            << "file to read " << fileRaw.data() << " file to write " << fileDataOut.data();
  mFileDest.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  mFileDest.open(fileRaw, std::fstream::in | std::fstream::binary);
  o2::fv0::LookUpTable lut{o2::fv0::ReadRaw::linear()};
  ReadRaw::readData(fileRaw.c_str(), lut);
  ReadRaw::writeDigits(fileDataOut.data());
}

void ReadRaw::readData(const std::string fileRaw, const o2::fv0::LookUpTable& lut)
{
  LOG(INFO) << " readData ";
  o2::header::RAWDataHeader mRDH;
  const char padding[CRUWordSize] = {0};
  int nStored = 0;

  o2::fv0::ChannelData chData;
  // get its size:
  mFileDest.seekg(0, mFileDest.end);
  long sizeFile = mFileDest.tellg();
  mFileDest.seekg(0);
  // read content of infile
  long posInFile = 0;

  while (posInFile < sizeFile - sizeof(mRDH)) {
    int pos = 0;
    mFileDest.seekg(posInFile);
    mFileDest.read(reinterpret_cast<char*>(&mRDH), sizeof(mRDH));
    //printRDH(&mRDH);
    int nwords = mRDH.memorySize;
    // LOG(INFO)<<"RDH Size "<<nwords;
    int npackages = mRDH.packetCounter;
    int numPage = mRDH.pageCnt;
    int offset = mRDH.offsetToNext;
    int link = mRDH.linkID;

    if (nwords <= sizeof(mRDH)) {
      posInFile += mRDH.offsetToNext;
      //LOG(INFO) << " next RDH  "<<"("<<rdh<<")" << posInFile;
      pos = 0;
    } else {
      posInFile += offset;
      pos = int(sizeof(mRDH));

      while (pos < nwords) {
        mFileDest.read(reinterpret_cast<char*>(&mEventHeader), sizeof(mEventHeader));
        LOG(INFO) << "Position " << pos << " nWords " << nwords << "  interac " << mEventHeader.orbit << "\n";
        pos += sizeof(mEventHeader);
        //LOG(DEBUG) << "read  header for link: " << link << " word: " << (int)mEventHeader.nGBTWords << " orbit: " << int(mEventHeader.orbit) << " BC: " << int(mEventHeader.bc) << " pos " << pos << " posinfile " << posInFile;
        o2::InteractionRecord intrec{uint16_t(mEventHeader.bc), uint32_t(mEventHeader.orbit)};

        if (link == LinkTCM) { //link for tcm
          mFileDest.read(reinterpret_cast<char*>(&mTCMdata), sizeof(mTCMdata));
          pos += sizeof(mTCMdata);
          LOG(INFO) << "read TCM  " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " BC " << int(mEventHeader.bc) << " pos " << pos << " posinfile " << posInFile;
        } else {
          if (mIsPadded) {
            pos += CRUWordSize - o2::fv0::EventHeader::PayloadSize;
          }
          for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
            //LOG(INFO)<< "NGBT WORDS  "<<mEventHeader.nGBTWords;
            mFileDest.read(reinterpret_cast<char*>(&mEventData[2 * i]), o2::fv0::EventData::PayloadSizeFirstWord);
            chData = {Short_t(lut.getChannel(link, (mEventData[2 * i].channelID))), Float_t(mEventData[2 * i].time), Short_t(mEventData[2 * i].charge)};
            mDigitAccum[intrec].emplace_back(chData);
            pos += o2::fv0::EventData::PayloadSizeFirstWord;

            /* LOG(INFO) << " read 1st word channelID " << int(mEventData[2 * i].channelID) << " charge " << mEventData[2 * i].charge
            << " time " << float(mEventData[2 * i].time) << " PM (crulink) " << link
            << " lut channel " << lut.getChannel(link, int(mEventData[2 * i].channelID)) << " pos " << pos<<"\n";
            */
            mFileDest.read(reinterpret_cast<char*>(&mEventData[2 * i + 1]), EventData::PayloadSizeSecondWord);
            pos += o2::fv0::EventData::PayloadSizeSecondWord;

            /*LOG(INFO) << " read 2nd word channelID " << int(mEventData[2 * i + 1].channelID)
            << " charge " << int(mEventData[2 * i + 1].charge) << " time " << float(mEventData[2 * i + 1].time)
            << " PM (crulink) " << link << " lut channel " << lut.getChannel(link, int(mEventData[2 * i + 1].channelID)) << " pos " << pos<<"\n";
            */

            chData = {Short_t(lut.getChannel(link, (mEventData[2 * i + 1].channelID))), Float_t(mEventData[2 * i + 1].time), Short_t(mEventData[2 * i + 1].charge)};
            mDigitAccum[intrec].emplace_back(chData);
            //LOG(INFO)<<"Interaction Record: "<< intrec<<"\n";
            //if (intrec.orbit>10000) LOG(INFO)<<"Interaction: "<< lut.getChannel(link, int(mEventData[2 * i + 1].channelID))<<"\n";
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
  std::vector<o2::fv0::ChannelData> chDataVecTree;
  std::vector<o2::fv0::BCData> chBcVecTree;

  for (auto& digit : mDigitAccum) {
    size_t nstored = 0;
    size_t first = chDataVecTree.size();
    LOG(INFO) << "Interaction Record: " << digit.first;
    for (auto& sec : digit.second) {
      chDataVecTree.emplace_back(int(sec.pmtNumber), float(sec.time), Short_t(sec.chargeAdc));
      nstored++;
    }
    chBcVecTree.emplace_back(first, nstored, digit.first);
    //LOG(INFO)<<"first "<< first << " nstored " <<nstored<<std::endl;
  }

  mOutTree->Branch("FV0DigitBC", &chBcVecTree);
  mOutTree->Branch("FV0DigitCh", &chDataVecTree);
  mOutTree->Fill();

  mOutFile->cd();
  mOutTree->Write();
  mOutFile->Close();
}
