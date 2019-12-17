// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFBase/WindowFiller.h"

#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>
#include "FairLogger.h"

using namespace o2::tof;

ClassImp(WindowFiller);

// How data acquisition works in real data
/*
           |<----------- 1 orbit ------------->|
     ------|-----------|-----------|-----------|------
             ^           ^           ^           ^ when triggers happen
        |<--- latency ---|
        |<- matching1->|
                    |<- matching2->|
                                |<- matching3->|
                                |<>| = overlap between two consecutive matching

Norbit = number of orbits elapsed
Nbunch = bunch in the current orbit (0:3563)
Ntdc = number of tdc counts within the matching window --> for 1/3 orbit (0:3649535)

raw time = trigger time (Norbit and Nbunch) - latency window + TDC(Ntdc)
 */

// What we implemented here (so far)
/*
           |<----------- 1 orbit ------------->|
     ------|-----------|-----------|-----------|------
           |<- matching1->|
                       |<- matching2->|
                                   |<- matching3->|
                                   |<>| = overlap between two consecutive matching windows

- NO OVERLAP between two consecutive windows at the moment (to be implemented)
- NO LATENCY WINDOW (we manage it during raw encoding/decoding) then digits already corrected

NBC = Number of bunch since timeframe beginning = Norbit*3564 + Nbunch
Ntdc = here within the current BC -> (0:1023)

digit time = NBC*1024 + Ntdc
 */

void WindowFiller::initObj()
{

  // method to initialize the parameters neede to digitize and the array of strip objects containing
  // the digits belonging to a strip

  for (Int_t i = 0; i < Geo::NSTRIPS; i++) {
    for (Int_t j = 0; j < MAXWINDOWS; j++) {
      mStrips[j].emplace_back(i);
      if (j < MAXWINDOWS - 1) {
        mStripsNext[j] = &(mStrips[j + 1]);
      }
    }
  }
}
//______________________________________________________________________
void WindowFiller::fillDigitsInStrip(std::vector<Strip>* strips, int channel, int tdc, int tot, int nbc, UInt_t istrip)
{
  (*strips)[istrip].addDigit(channel, tdc, tot * Geo::NTOTBIN_PER_NS, nbc);
}
//______________________________________________________________________
void WindowFiller::fillOutputContainer(std::vector<Digit>& digits)
{
  if (mContinuous) {
    digits.clear();
  //   mMCTruthOutputContainer->clear();
  }

  printf("TOF fill output container\n");
  // filling the digit container doing a loop on all strips
  for (auto& strip : *mStripsCurrent) {
    strip.fillOutputContainer(digits);
  }

  if (mContinuous) {
    printf("%i) # TOF digits = %lu (%p)\n", mIcurrentReadoutWindow, digits.size(), mStripsCurrent);
    mDigitsPerTimeFrame.push_back(digits);
  }

  // copying the transient labels to the output labels (stripping the tdc information)
  // if (mMCTruthOutputContainer) {
  //   // copy from transientTruthContainer to mMCTruthAray
  //   // a brute force solution for the moment; should be handled by a dedicated API
  //   for (int index = 0; index < mMCTruthContainerCurrent->getIndexedSize(); ++index) {
  //     mMCTruthOutputContainer->addElements(index, mMCTruthContainerCurrent->getLabels(index));
  //   }
  // }

  // if (mContinuous)
  //   mMCTruthOutputContainerPerTimeFrame.push_back(*mMCTruthOutputContainer);
  // mMCTruthContainerCurrent->clear();

  // switch to next mStrip after flushing current readout window data
  mIcurrentReadoutWindow++;
  if (mIcurrentReadoutWindow >= MAXWINDOWS)
    mIcurrentReadoutWindow = 0;
  mStripsCurrent = &(mStrips[mIcurrentReadoutWindow]);
  // mMCTruthContainerCurrent = &(mMCTruthContainer[mIcurrentReadoutWindow]);
  int k = mIcurrentReadoutWindow + 1;
  for (Int_t i = 0; i < MAXWINDOWS - 1; i++) {
    if (k >= MAXWINDOWS)
      k = 0;
    // mMCTruthContainerNext[i] = &(mMCTruthContainer[k]);
    mStripsNext[i] = &(mStrips[k]);
    k++;
  }
}
//______________________________________________________________________
void WindowFiller::flushOutputContainer(std::vector<Digit>& digits)
{ // flush all residual buffered data
  // TO be implemented
  printf("flushOutputContainer\n");
  if (!mContinuous)
    fillOutputContainer(digits);
  else {
    for (Int_t i = 0; i < MAXWINDOWS; i++) {
      fillOutputContainer(digits); // fill all windows which are before (not yet stored) of the new current one
      checkIfReuseFutureDigits();
      mReadoutWindowCurrent++;
    }

    while (mFutureDigits.size()) {
      fillOutputContainer(digits); // fill all windows which are before (not yet stored) of the new current one
      checkIfReuseFutureDigits();
      mReadoutWindowCurrent++;
    }
  }
}
//______________________________________________________________________
void WindowFiller::checkIfReuseFutureDigits()
{
  // check if digits stored very far in future match the new readout windows currently available
  int idigit = mFutureDigits.size() - 1;

  int bclimit = 999999; // if bc is larger than this value stop the search  in the next loop since bc are ordered in descending order

  for (std::vector<Digit>::reverse_iterator digit = mFutureDigits.rbegin(); digit != mFutureDigits.rend(); ++digit) {

    if(digit->getBC() > bclimit) break;

    double timestamp = digit->getBC() * Geo::BC_TIME + digit->getTDC() * Geo::TDCBIN * 1E-3;        // in ns
    int isnext = Int_t(timestamp * Geo::READOUTWINDOW_INV) - (mReadoutWindowCurrent + 1); // to be replaced with uncalibrated time
 
    if (isnext < 0){                                                           // we jump too ahead in future, digit will be not stored
      LOG(INFO) << "Digit lost because we jump too ahead in future. Current RO window=" << isnext << "\n";
      
      // remove digit from array in the future
      int labelremoved = digit->getLabel();
      mFutureDigits.erase(mFutureDigits.begin() + idigit);
      
      idigit--;
      
      continue;
    }
    
    
    if (isnext < MAXWINDOWS - 1) { // move from digit buffer array to the proper window
      std::vector<Strip>* strips = mStripsCurrent;
      // o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mcTruthContainer = mMCTruthContainerCurrent;

      if (isnext) {
	strips = mStripsNext[isnext - 1];
	// mcTruthContainer = mMCTruthContainerNext[isnext - 1];
      }

      // int trackID = mFutureItrackID[digit->getLabel()];
      // int sourceID = mFutureIsource[digit->getLabel()];
      // int eventID = mFutureIevent[digit->getLabel()];
      // fillDigitsInStrip(strips, mcTruthContainer, digit->getChannel(), digit->getTDC(), digit->getTOT(), digit->getBC(), digit->getChannel() / Geo::NPADS, trackID, eventID, sourceID);
      fillDigitsInStrip(strips, digit->getChannel(), digit->getTDC(), digit->getTOT(), digit->getBC(), digit->getChannel() / Geo::NPADS);
      
      // // remove the element from the buffers
      // mFutureItrackID.erase(mFutureItrackID.begin() + digit->getLabel());
      // mFutureIsource.erase(mFutureIsource.begin() + digit->getLabel());
      // mFutureIevent.erase(mFutureIevent.begin() + digit->getLabel());

      // int labelremoved = digit->getLabel();
      mFutureDigits.erase(mFutureDigits.begin() + idigit);

      // // adjust labels
      // for (auto& digit2 : mFutureDigits) {
      //   if (digit2.getLabel() > labelremoved) {
      //     digit2.setLabel(digit2.getLabel() - 1);
      //   }
      // }
      // remove also digit from buffer
      //      mFutureDigits.erase(mFutureDigits.begin() + idigit);
    }
    else{
      bclimit = digit->getBC();
    }
    idigit--; // go back to the next position in the reverse iterator
  }           // close future digit loop
}
