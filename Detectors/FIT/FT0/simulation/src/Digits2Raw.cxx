// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/RawEventData.h"
#include "FT0Simulation/Digits2Raw.h"
#include "DetectorsBase/Triggers.h"
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
//Digits2Raw(char * fileRaw, std::string fileDigitsName)
void Digits2Raw::readDigits(const char* fileRaw, const char* fileDigitsName)
{
  std::cout << "**********Digits2Raw::convertDigits" << std::endl;

  mFileDest.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  mFileDest.open(fileRaw);
  // mFileDest = fopen(fileRaw, "wb+");
  o2::ft0::LookUpTable lut{o2::ft0::Digits2Raw::linear()};
  std::cout << " ##### LookUp set " << std::endl;

  TFile* fdig = TFile::Open(fileDigitsName);
  assert(fdig != nullptr);
  std::cout << " Open digits file " << std::endl;
  TTree* digTree = (TTree*)fdig->Get("o2sim");
  std::vector<o2::ft0::Digit>* digArr = new std::vector<o2::ft0::Digit>;
  digTree->SetBranchAddress("FT0Digit", &digArr);
  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  std::cout << "Found " << nevD << " events with digits " << std::endl;
  uint32_t old_orbit = 0;
  for (Int_t iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
    for (const auto& digit : *digArr) {

      auto mIntRecord = digit.getInteractionRecord();
      uint32_t current_orbit = mIntRecord.orbit;
      if (old_orbit != current_orbit) {
        old_orbit = current_orbit;
        mNpages = 0;
      }
      convertDigits(digit, lut);
    }
  }
}

/*******************************************************************************************************************/
void Digits2Raw::convertDigits(const o2::ft0::Digit& digit, const o2::ft0::LookUpTable& lut)
{
  auto mIntRecord = digit.getInteractionRecord();
  std::vector<o2::ft0::ChannelData> mTimeAmp = digit.getChDgData();
  bool is0TVX = digit.getisVrtx();
  int oldlink = -1;
  int nchannels = 0;
  for (auto& d : mTimeAmp) {
    int nlink = lut.getLink(d.ChId);
    if (nlink != oldlink) {
      uint nGBTWords = uint((nchannels + 1) / 2);
      flushEvent(oldlink, mIntRecord, nGBTWords);
      oldlink = nlink;
      nchannels = 0;
      mNpages++;
    }
    mEventData[nchannels].channelID = lut.getMCP(d.ChId);
    mEventData[nchannels].charge = MV_2_NCHANNELS * d.QTCAmpl;   //7 mV ->16channels
    mEventData[nchannels].time = CFD_NS_2_NCHANNELS * d.CFDTime; //1000.(ps)/13.2(channel);
    mEventData[nchannels].is1TimeLostEvent = 0;
    mEventData[nchannels].is2TimeLostEvent = 0;
    mEventData[nchannels].isADCinGate = 1;
    mEventData[nchannels].isAmpHigh = 0;
    mEventData[nchannels].isDoubleEvent = 0;
    mEventData[nchannels].isEventInTVDC = is0TVX ? 1 : 0;
    mEventData[nchannels].isTimeInfoLate = 0;
    mEventData[nchannels].isTimeInfoLost = 0;
    int chain = std::rand() % 2;
    mEventData[nchannels].numberADC = chain ? 1 : 0;
    //  std::cout << "@@@@ packed GBT " << nlink << " channelID   " << mEventData[nchannels].channelID << " charge " << mEventData[nchannels].charge << " time " << mEventData[nchannels].time << " chain " << mEventData[nchannels].numberADC << std::endl;
    //  std::cout << "@@@@ digits channelID   " << d.ChId  << " charge " << d.QTCAmpl << " time " << d.CFDTime << std::endl;
    nchannels++;
  }
  if ((nchannels % 2) == 1)
    mEventData[nchannels] = {};
}

void Digits2Raw::flushEvent(int link, o2::InteractionRecord const& mIntRecord, uint nGBTWords)
{
  // If we are called before the first link, just exit
  if (link < 0)
    return;
  setRDH(link, mIntRecord, nGBTWords);
  setGBTHeader(link, mIntRecord, nGBTWords);
  mFileDest.write(reinterpret_cast<char*>(&mRDH), sizeof(mRDH));
  mFileDest.write(reinterpret_cast<char*>(&mEventHeader), sizeof(mEventHeader));
  mFileDest.write(reinterpret_cast<char*>(&mEventData), sizeof(mEventData));
}
//_____________________________________________________________________________________
void Digits2Raw::setGBTHeader(int link, o2::InteractionRecord const& mIntRecord, uint nGBTWords)
{
  mEventHeader.startDescriptor = 15;
  mEventHeader.Nchannels = nGBTWords;
  mEventHeader.reservedField = 0;
  mEventHeader.bc = mIntRecord.bc;
  mEventHeader.orbit = mIntRecord.orbit;
}
//_____________________________________________________________________________
void Digits2Raw::setRDH(int nlink, o2::InteractionRecord const& mIntRecord, uint nGBTWords)
{
  mRDH.triggerOrbit = mRDH.heartbeatOrbit = mIntRecord.orbit;
  mRDH.triggerBC = mRDH.heartbeatBC = mIntRecord.bc;
  mRDH.linkID = nlink;
  mRDH.feeId = nlink;

  mRDH.triggerType = o2::trigger::PhT; // ??
  mRDH.detectorField = 0xffff;         //empty for FIt yet
  mRDH.blockLength = 0xffff;           // ITS keeps this dummy
  mRDH.stop = 0;
  mRDH.memorySize = mRDH.headerSize + (nGBTWords + 1) * GBTWORDSIZE; // update remaining size
  mRDH.pageCnt = mNpages;
  printRDH(&mRDH);
}
//_____________________________________________________________________________
void Digits2Raw::printRDH(const o2::header::RAWDataHeader* h)
{
  if (!h) {
    printf("Provided RDH pointer is null\n");
    return;
  }
  printf("RDH| Ver:%2u Hsz:%2u Blgt:%4u FEEId:0x%04x PBit:%u\n",
         uint32_t(h->version), uint32_t(h->headerSize), uint32_t(h->blockLength), uint32_t(h->feeId), uint32_t(h->priority));
  printf("RDH|[CRU: Offs:%5u Msz:%4u LnkId:0x%02x Packet:%3u CRUId:0x%04x]\n",
         uint32_t(h->offsetToNext), uint32_t(h->memorySize), uint32_t(h->linkID), uint32_t(h->packetCounter), uint32_t(h->cruID));
  printf("RDH| TrgOrb:%9u HBOrb:%9u TrgBC:%4u HBBC:%4u TrgType:%u\n",
         uint32_t(h->triggerOrbit), uint32_t(h->heartbeatOrbit), uint32_t(h->triggerBC), uint32_t(h->heartbeatBC),
         uint32_t(h->triggerType));
  printf("RDH| DetField:0x%05x Par:0x%04x Stop:0x%04x PageCnt:%5u\n",
         uint32_t(h->detectorField), uint32_t(h->par), uint32_t(h->stop), uint32_t(h->pageCnt));
}
