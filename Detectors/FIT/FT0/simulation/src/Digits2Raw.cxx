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
  std::vector<o2::ft0::Topo> lut_data(NCHANNELS_PM * NPMs);
  for (int link = 0; link < NPMs; ++link)
    for (int quadrant = 0; quadrant < NCHANNELS_PM; ++quadrant)
      lut_data[link * NCHANNELS_PM + quadrant] = o2::ft0::Topo{link, quadrant};
  o2::ft0::LookUpTable lut{lut_data};

  std::cout << " ##### LookUp set " << std::endl;
  mFileDest.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  mFileDest.open(fileRaw);
  // mFileDest = fopen(fileRaw, "wb+");

  TFile* fdig = TFile::Open(fileDigitsName);
  assert(fdig != nullptr);
  std::cout << " Open digits file " << std::endl;
  TTree* digTree = (TTree*)fdig->Get("o2sim");
  std::vector<o2::ft0::Digit>* digArr = new std::vector<o2::ft0::Digit>;
  digTree->SetBranchAddress("FT0Digit", &digArr);
  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  std::cout << "Found " << nevD << " events with digits " << std::endl;
  for (Int_t iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
    for (const auto& digit : *digArr) {
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
  int nlink = 0;
  int nchannels = 0;
  for (auto& d : mTimeAmp) {
    nlink = lut.getLink(d.ChId);
    if (nlink != oldlink) {
      flushEvent(oldlink, mIntRecord, nchannels);
      oldlink = nlink;
      nchannels = 0;
    }
    mEventData[nchannels].channelID = lut.getQuadrant(d.ChId);
    mEventData[nchannels].charge = 2.2857143 * d.QTCAmpl; //7 mV ->16channels
    mEventData[nchannels].time = 75.757576 * d.CFDTime;   //1000.(ps)/13.2(channel);
    //   std::cout << "@@@@ packed GBT "<<nlink<<" channelID   " << mEventData[nchannels].channelID << " charge " << mEventData[nchannels].charge << " time " << mEventData[nchannels].time << std::endl;
    //   std::cout << "@@@@ digits channelID   " << d.ChId  << " charge " << d.QTCAmpl << " time " << d.CFDTime << std::endl;
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
    //   std::cout << "@@@@ packed GBT "<<nlink<<" channelID   " << mEventData[nchannels].channelID << " charge " << mEventData[nchannels].charge << " time " << mEventData[nchannels].time << std::endl;
    //   std::cout << "@@@@ digits channelID   " << d.ChId  << " charge " << d.QTCAmpl << " time " << d.CFDTime << std::endl;
    nchannels++;
  }
  flushEvent(oldlink, mIntRecord, nchannels);
}

void Digits2Raw::flushEvent(int link, o2::InteractionRecord const& mIntRecord, uint nchannels)
{
  // If we are called before the first link, just exit
  if (link < 0)
    return;
  std::cout << " new GBT " << link << " with N ch " << nchannels << std::endl;
  setRDH(link, mIntRecord);
  setGBTHeader(link, mIntRecord, nchannels);
  // mFileDest.write(reinterpret_cast<char*>(&mRDH), sizeof(mRDH));
  mFileDest.write(reinterpret_cast<char*>(&mEventHeader), sizeof(mEventHeader));
  mFileDest.write(reinterpret_cast<char*>(&mEventData), sizeof(mEventData));
}
//_____________________________________________________________________________________
void Digits2Raw::setGBTHeader(int link, o2::InteractionRecord const& mIntRecord, uint nchannels)
{
  mEventHeader.startDescriptor = 15;
  mEventHeader.Nchannels = nchannels;
  mEventHeader.reservedField = 0;
  mEventHeader.bc = mIntRecord.bc;
  mEventHeader.orbit = mIntRecord.orbit;
}
//_____________________________________________________________________________
void Digits2Raw::setRDH(int nlink, o2::InteractionRecord const& mIntRecord)
{
  mRDH.triggerOrbit = mRDH.heartbeatOrbit = mIntRecord.orbit;
  mRDH.triggerBC = mRDH.heartbeatBC = mIntRecord.bc;
  mRDH.linkID = nlink;
  std::cout << "### setRDH link " << nlink << " orbit " << mRDH.triggerOrbit << " BC " << mRDH.triggerBC << std::endl;
}
