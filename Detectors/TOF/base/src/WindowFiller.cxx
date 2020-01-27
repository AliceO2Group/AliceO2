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
void WindowFiller::fillDigitsInStrip(std::vector<Strip>* strips, int channel, int tdc, int tot, int nbc, UInt_t istrip, Int_t triggerorbit, Int_t triggerbunch)
{
  (*strips)[istrip].addDigit(channel, tdc, tot * Geo::NTOTBIN_PER_NS, nbc, 0, triggerorbit, triggerbunch);
}
//______________________________________________________________________
void WindowFiller::fillOutputContainer(std::vector<Digit>& digits)
{
  if (mContinuous) {
    digits.clear();
  }

  // filling the digit container doing a loop on all strips
  for (auto& strip : *mStripsCurrent) {
    strip.fillOutputContainer(digits);
  }

  if (mContinuous) {
    int first = mDigitsPerTimeFrame.size();
    int ne = digits.size();
    ReadoutWindowData info(first, ne);
    if (digits.size())
      mDigitsPerTimeFrame.insert(mDigitsPerTimeFrame.end(), digits.begin(), digits.end());
    mReadoutWindowData.push_back(info);
  }

  // switch to next mStrip after flushing current readout window data
  mIcurrentReadoutWindow++;
  if (mIcurrentReadoutWindow >= MAXWINDOWS)
    mIcurrentReadoutWindow = 0;
  mStripsCurrent = &(mStrips[mIcurrentReadoutWindow]);
  int k = mIcurrentReadoutWindow + 1;
  for (Int_t i = 0; i < MAXWINDOWS - 1; i++) {
    if (k >= MAXWINDOWS)
      k = 0;
    mStripsNext[i] = &(mStrips[k]);
    k++;
  }

  mReadoutWindowCurrent++;
}
//______________________________________________________________________
void WindowFiller::flushOutputContainer(std::vector<Digit>& digits)
{ // flush all residual buffered data
  // TO be implemented

  printf("flushOutputContainer\n");
  for (Int_t i = 0; i < MAXWINDOWS; i++) {
    int n = 0;
    for (int j = 0; j < mStrips[i].size(); j++)
      n += ((mStrips[i])[j]).getNumberOfDigits();

    printf("ro #%d: digits = %d\n", i, n);
  }

  printf("Future digits = %lu\n", mFutureDigits.size());

  if (!mContinuous)
    fillOutputContainer(digits);
  else {
    for (Int_t i = 0; i < MAXWINDOWS; i++) {
      fillOutputContainer(digits); // fill all windows which are before (not yet stored) of the new current one
      checkIfReuseFutureDigitsRO();
    }

    int round = 0;
    while (mFutureDigits.size()) {
      round++;
      fillOutputContainer(digits); // fill all windows which are before (not yet stored) of the new current one
      checkIfReuseFutureDigitsRO();
    }

    for (Int_t i = 0; i < MAXWINDOWS; i++) {
      fillOutputContainer(digits); // fill last readout windows
    }
  }
}
//______________________________________________________________________
void WindowFiller::checkIfReuseFutureDigits()
{
  if (!mFutureDigits.size())
    return;

  // check if digits stored very far in future match the new readout windows currently available
  if (mFutureToBeSorted) {
    // sort digit in descending BC order: kept last as first
    std::sort(mFutureDigits.begin(), mFutureDigits.end(),
              [](o2::tof::Digit a, o2::tof::Digit b) { return a.getBC() > b.getBC(); });
    mFutureToBeSorted = false;
  }

  int idigit = mFutureDigits.size() - 1;

  int bclimit = 999999; // if bc is larger than this value stop the search  in the next loop since bc are ordered in descending order

  for (std::vector<Digit>::reverse_iterator digit = mFutureDigits.rbegin(); digit != mFutureDigits.rend(); ++digit) {

    if (digit->getBC() > bclimit)
      break;

    double timestamp = digit->getBC() * Geo::BC_TIME + digit->getTDC() * Geo::TDCBIN * 1E-3; // in ns
    int isnext = Int_t(timestamp * Geo::READOUTWINDOW_INV) - (mReadoutWindowCurrent + 1);    // to be replaced with uncalibrated time

    if (isnext < 0) { // we jump too ahead in future, digit will be not stored
      LOG(INFO) << "Digit lost because we jump too ahead in future. Current RO window=" << isnext << "\n";

      // remove digit from array in the future
      int labelremoved = digit->getLabel();
      mFutureDigits.erase(mFutureDigits.begin() + idigit);

      idigit--;

      continue;
    }

    if (isnext < MAXWINDOWS - 1) { // move from digit buffer array to the proper window
      std::vector<Strip>* strips = mStripsCurrent;

      if (isnext) {
        strips = mStripsNext[isnext - 1];
      }

      fillDigitsInStrip(strips, digit->getChannel(), digit->getTDC(), digit->getTOT(), digit->getBC(), digit->getChannel() / Geo::NPADS);

      // int labelremoved = digit->getLabel();
      mFutureDigits.erase(mFutureDigits.begin() + idigit);

    } else {
      bclimit = digit->getBC();
    }
    idigit--; // go back to the next position in the reverse iterator
  }           // close future digit loop
}
//______________________________________________________________________
void WindowFiller::checkIfReuseFutureDigitsRO() // the same but using readout info information from raw
{
  if (!mFutureDigits.size())
    return;

  // check if digits stored very far in future match the new readout windows currently available
  if (mFutureToBeSorted) {
    // sort digit in descending BC order: kept last as first
    std::sort(mFutureDigits.begin(), mFutureDigits.end(),
              [](o2::tof::Digit a, o2::tof::Digit b) {
                if (a.getTriggerOrbit() != b.getTriggerOrbit())
                  return a.getTriggerOrbit() > b.getTriggerOrbit();
                if (a.getTriggerBunch() != b.getTriggerBunch())
                  return a.getTriggerBunch() > b.getTriggerBunch();
                return a.getBC() > b.getBC();
              });
    mFutureToBeSorted = false;
  }

  int idigit = mFutureDigits.size() - 1;

  int rolimit = 999999; // if bc is larger than this value stop the search  in the next loop since bc are ordered in descending order

  for (std::vector<Digit>::reverse_iterator digit = mFutureDigits.rbegin(); digit != mFutureDigits.rend(); ++digit) {

    int row = (digit->getTriggerOrbit() - mFirstOrbit) * Geo::BC_IN_ORBIT + (digit->getTriggerBunch() - mFirstBunch) + 100; // N bunch id of the trigger from timeframe start + 100 bunches

    row *= Geo::BC_IN_WINDOW_INV;

    if (row > rolimit)
      break;

    int isnext = row - mReadoutWindowCurrent;

    if (isnext < 0) { // we jump too ahead in future, digit will be not stored
      LOG(INFO) << "Digit lost because we jump too ahead in future. Current RO window=" << isnext << "\n";

      // remove digit from array in the future
      int labelremoved = digit->getLabel();
      mFutureDigits.erase(mFutureDigits.begin() + idigit);

      idigit--;

      continue;
    }

    if (isnext < MAXWINDOWS - 1) { // move from digit buffer array to the proper window
      std::vector<Strip>* strips = mStripsCurrent;

      if (isnext) {
        strips = mStripsNext[isnext - 1];
      }

      fillDigitsInStrip(strips, digit->getChannel(), digit->getTDC(), digit->getTOT(), digit->getBC(), digit->getChannel() / Geo::NPADS);

      // int labelremoved = digit->getLabel();
      mFutureDigits.erase(mFutureDigits.begin() + idigit);
    } else {
      rolimit = row;
    }
    idigit--; // go back to the next position in the reverse iterator
  }           // close future digit loop
}
