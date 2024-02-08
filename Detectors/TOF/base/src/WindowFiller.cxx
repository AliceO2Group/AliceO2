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
#include <fairlogger/Logger.h>
#include "DataFormatsTOF/CompressedDataFormat.h"

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
void WindowFiller::reset()
{
  mIcurrentReadoutWindow = 0;
  mReadoutWindowCurrent = 0;

  for (Int_t i = 0; i < MAXWINDOWS; i++) {
    for (Int_t j = 0; j < Geo::NSTRIPS; j++) {
      mStrips[i][j].clear();
    }
  }
  mFutureDigits.clear();

  mStripsCurrent = &(mStrips[0]);
  mStripsNext[0] = &(mStrips[1]);

  mDigitsPerTimeFrame.clear();
  mReadoutWindowData.clear();
  mReadoutWindowDataFiltered.clear();

  mDigitHeader.clear();

  mFirstIR.bc = 0;
  mFirstIR.orbit = 0;
}

//______________________________________________________________________
void WindowFiller::fillDigitsInStrip(std::vector<Strip>* strips, int channel, int tdc, int tot, uint64_t nbc, UInt_t istrip, uint32_t triggerorbit, uint16_t triggerbunch)
{
  if (channel > -1) { // check channel validity
    (*strips)[istrip].addDigit(channel, tdc, tot, nbc, 0, triggerorbit, triggerbunch);
  }
}
//______________________________________________________________________
void WindowFiller::addCrateHeaderData(unsigned long orbit, int crate, int32_t bc, uint32_t eventCounter)
{
  if (orbit < mFirstIR.orbit) {
    return;
  }
  orbit -= mFirstIR.orbit;

  orbit *= Geo::NWINDOW_IN_ORBIT;          // move from orbit to N readout window
  orbit += (bc + 100) / Geo::BC_IN_WINDOW; // select readout window in the orbit according to the BC (100 shift to avoid border effects)

  if (mCrateHeaderData.size() < orbit + 1) {
    mCrateHeaderData.resize(orbit + 1);
  }

  mCrateHeaderData[orbit].bc[crate] = bc;
  mCrateHeaderData[orbit].eventCounter[crate] = eventCounter;
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
    int orbit_shift = mReadoutWindowData.size() / Geo::NWINDOW_IN_ORBIT;

    mDigitHeader.addRow();

    int bc_shift = -1;
    int eventcounter = -1;
    int ncratesSeen = 0;
    if (mReadoutWindowData.size() >= mCrateHeaderData.size()) {
      bc_shift = (mReadoutWindowData.size() % Geo::NWINDOW_IN_ORBIT) * Geo::BC_IN_WINDOW; // insert default value
      eventcounter = mReadoutWindowData.size() % 4096;
      for (int icrate = 0; icrate < Geo::kNCrate; icrate++) {
        info.setEmptyCrate(icrate);
      }
    } else {
      unsigned long irow = mReadoutWindowData.size();
      for (int icrate = 0; icrate < Geo::kNCrate; icrate++) {
        if (mCrateHeaderData[irow].bc[icrate] == -1) { // crate not read
          info.setEmptyCrate(icrate);
          continue;
        } else {
          mDigitHeader.crateSeen(icrate);
        }
        ncratesSeen++;

        if (bc_shift == -1 || mCrateHeaderData[irow].bc[icrate] < bc_shift) {
          bc_shift = mCrateHeaderData[irow].bc[icrate];
        }
        if (eventcounter == -1 || mCrateHeaderData[irow].eventCounter[icrate] < eventcounter) {
          eventcounter = mCrateHeaderData[irow].eventCounter[icrate];
        }
      }

      mDigitHeader.numCratesSeen(ncratesSeen);

      if (bc_shift == -1) {
        bc_shift = (mReadoutWindowData.size() % Geo::NWINDOW_IN_ORBIT) * Geo::BC_IN_WINDOW; // insert default value
      }
      if (eventcounter == -1) {
        eventcounter = mReadoutWindowData.size() % 4096; // insert default value
      }
    }

    info.setBCData(mFirstIR.orbit + orbit_shift, mFirstIR.bc + bc_shift);
    info.setEventCounter(eventcounter);
    int firstPattern = mPatterns.size();
    int npatterns = 0;

    // check if patterns are in the current row
    unsigned int initrow = mFirstIR.orbit * Geo::NWINDOW_IN_ORBIT;
    for (std::vector<PatternData>::reverse_iterator it = mCratePatterns.rbegin(); it != mCratePatterns.rend(); ++it) {
      //printf("pattern row=%ld current=%ld\n",it->row - initrow,mReadoutWindowCurrent);

      if (it->row - initrow > mReadoutWindowCurrent) {
        break;
      }

      if (it->row - initrow < mReadoutWindowCurrent) { // this should not happen
        LOG(error) << "One pattern skipped because appears to occur early of the current row " << it->row << " < " << mReadoutWindowCurrent << " ?!";
      } else {
        uint32_t cpatt = it->pattern;
        auto dpatt = reinterpret_cast<compressed::Diagnostic_t*>(&cpatt);
        uint8_t slot = dpatt->slotID;
        uint32_t cbit = 1;

        mPatterns.push_back(slot + 28); // add slot
        info.addedDiagnostic(it->icrate);
        npatterns++;

        for (int ibit = 0; ibit < 28; ibit++) {
          if (dpatt->faultBits & cbit) {
            mPatterns.push_back(ibit); // add bit error
            info.addedDiagnostic(it->icrate);
            npatterns++;
          }
          cbit <<= 1;
        }
        //        uint8_t w1 = cpatt & 0xff;
        //        uint8_t w2 = (cpatt >> 8) & 0xff;
        //        uint8_t w3 = (cpatt >> 16) & 0xff;
        //        uint8_t w4 = (cpatt >> 24) & 0xff;
        ////      cpatt = w1 + (w2 + (w3 + uint(w4)*256)*256)*256;
        //        mPatterns.push_back(w1);
        //        info.addedDiagnostic(it->icrate);
        //        npatterns++;
        //        mPatterns.push_back(w2);
        //        info.addedDiagnostic(it->icrate);
        //        npatterns++;
        //        mPatterns.push_back(w3);
        //        info.addedDiagnostic(it->icrate);
        //        npatterns++;
        //        mPatterns.push_back(w4);
        //        info.addedDiagnostic(it->icrate);
        //        npatterns++;
      }
      mCratePatterns.pop_back();
    }

    info.setFirstEntryDia(firstPattern);
    info.setNEntriesDia(npatterns);
    if (digits.size() || npatterns) {
      mDigitsPerTimeFrame.insert(mDigitsPerTimeFrame.end(), digits.begin(), digits.end());
      mReadoutWindowDataFiltered.push_back(info);
    }
    mReadoutWindowData.push_back(info);
  }

  // switch to next mStrip after flushing current readout window data
  mIcurrentReadoutWindow++;
  if (mIcurrentReadoutWindow >= MAXWINDOWS) {
    mIcurrentReadoutWindow = 0;
  }
  mStripsCurrent = &(mStrips[mIcurrentReadoutWindow]);
  int k = mIcurrentReadoutWindow + 1;
  for (Int_t i = 0; i < MAXWINDOWS - 1; i++) {
    if (k >= MAXWINDOWS) {
      k = 0;
    }
    mStripsNext[i] = &(mStrips[k]);
    k++;
  }

  mReadoutWindowCurrent++;
}
//______________________________________________________________________
void WindowFiller::flushOutputContainer(std::vector<Digit>& digits)
{ // flush all residual buffered data
  // TO be implemented

  // sort patterns (diagnostic words) in time
  std::sort(mCratePatterns.begin(), mCratePatterns.end(),
            [](PatternData a, PatternData b) { if(a.row == b.row) { return a.icrate > b.icrate; } else { return a.row > b.row; } });

  for (Int_t i = 0; i < MAXWINDOWS; i++) {
    int n = 0;
    for (int j = 0; j < mStrips[i].size(); j++) {
      n += ((mStrips[i])[j]).getNumberOfDigits();
    }
  }

  checkIfReuseFutureDigitsRO();

  if (!mContinuous) {
    fillOutputContainer(digits);
  } else {
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

    int nwindowperTF = o2::tof::Utils::getNOrbitInTF() * Geo::NWINDOW_IN_ORBIT;

    for (Int_t i = 0; i < MAXWINDOWS; i++) {
      if (mReadoutWindowData.size() < nwindowperTF) {
        fillOutputContainer(digits); // fill last readout windows
      }
    }

    // check that all orbits are complete in terms of number of readout windows
    while ((mReadoutWindowData.size() % nwindowperTF)) {
      fillOutputContainer(digits); // fill windows without digits to complete all orbits in the last TF
    }
  }
}
//______________________________________________________________________
void WindowFiller::checkIfReuseFutureDigits()
{
  if (!mFutureDigits.size()) {
    return;
  }

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

    if (digit->getBC() > bclimit) {
      break;
    }

    double timestamp = digit->getBC() * Geo::BC_TIME + digit->getTDC() * Geo::TDCBIN * 1E-3; // in ns
    int isnext = Int_t(timestamp * Geo::READOUTWINDOW_INV) - (mReadoutWindowCurrent + 1);    // to be replaced with uncalibrated time

    if (isnext < 0) { // we jump too ahead in future, digit will be not stored
      LOG(debug) << "Digit lost because we jump too ahead in future. Current RO window=" << isnext << "\n";

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
  if (!mFutureDigits.size()) {
    return;
  }

  // check if digits stored very far in future match the new readout windows currently available
  if (mFutureToBeSorted) {
    // sort digit in descending BC order: kept last as first
    std::sort(mFutureDigits.begin(), mFutureDigits.end(),
              [](o2::tof::Digit a, o2::tof::Digit b) {
                if (a.getTriggerOrbit() != b.getTriggerOrbit()) {
                  return a.getTriggerOrbit() > b.getTriggerOrbit();
                }
                if (a.getTriggerBunch() != b.getTriggerBunch()) {
                  return a.getTriggerBunch() > b.getTriggerBunch();
                }
                return a.getBC() > b.getBC();
              });
    mFutureToBeSorted = false;
  }

  int idigit = mFutureDigits.size() - 1;

  int rolimit = 999999; // if bc is larger than this value stop the search  in the next loop since bc are ordered in descending order

  for (std::vector<Digit>::reverse_iterator digit = mFutureDigits.rbegin(); digit != mFutureDigits.rend(); ++digit) {

    int row = (digit->getTriggerOrbit() - mFirstIR.orbit) * Geo::BC_IN_ORBIT + (digit->getTriggerBunch() - mFirstIR.bc) + 100; // N bunch id of the trigger from timeframe start + 100 bunches

    row *= Geo::BC_IN_WINDOW_INV;

    if (row > rolimit) {
      break;
    }

    int isnext = row - mReadoutWindowCurrent;

    if (isnext < 0) { // we jump too ahead in future, digit will be not stored
      LOG(debug) << "Digit lost because we jump too ahead in future. Current RO window=" << isnext << "\n";

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

      if (mMaskNoiseRate < 0 || mChannelCounts[digit->getChannel()] < mMaskNoiseRate) {
        fillDigitsInStrip(strips, digit->getChannel(), digit->getTDC(), digit->getTOT(), digit->getBC(), digit->getChannel() / Geo::NPADS);
      }

      // int labelremoved = digit->getLabel();
      mFutureDigits.erase(mFutureDigits.begin() + idigit);
    } else {
      rolimit = row;
    }
    idigit--; // go back to the next position in the reverse iterator
  }           // close future digit loop
}

void WindowFiller::fillDiagnosticFrequency()
{
  bool isTOFempty = true;
  mDiagnosticFrequency.clear();
  // fill diagnostic frequency
  for (int j = 0; j < mReadoutWindowData.size(); j++) {
    mDiagnosticFrequency.fillROW();
    int fd = mReadoutWindowData[j].firstDia();
    for (int ic = 0; ic < 72; ic++) {
      if (ic) {
        fd += mReadoutWindowData[j].getDiagnosticInCrate(ic - 1);
      }
      int dia = mReadoutWindowData[j].getDiagnosticInCrate(ic);
      int slot = 0;
      if (mReadoutWindowData[j].isEmptyCrate(ic)) {
        mDiagnosticFrequency.fillEmptyCrate(ic);
      } else {
        isTOFempty = false;
        if (dia) {
          int lastdia = fd + dia;

          ULong64_t key;
          for (int dd = fd; dd < lastdia; dd++) {
            if (mPatterns[dd] >= 28) {
              slot = mPatterns[dd] - 28;
              key = mDiagnosticFrequency.getTRMKey(ic, slot);
              continue;
            }

            key += (1 << mPatterns[dd]);

            if (dd + 1 == lastdia || mPatterns[dd + 1] >= 28) {
              mDiagnosticFrequency.fill(key);
            }
          }
        }
      }
    }
  }

  if (isTOFempty) {
    mDiagnosticFrequency.fillEmptyTOF();
  }

  // fill also noise diagnostic if the counts within the TF is larger than a threashold (default >=11, -> 1 kHZ)
  int masknoise = mMaskNoiseRate;
  if (masknoise < 0) {
    masknoise = -masknoise;
  }

  for (int i = 0; i < Geo::NCHANNELS; i++) {
    if (mChannelCounts[i] >= masknoise) {
      int additionalMask = 0;

      if (mChannelCounts[i] >= masknoise * 10) {
        additionalMask += (1 << 19); // > 10 kHZ (if masknoise = 1 kHz)
        if (mChannelCounts[i] >= masknoise * 100) {
          additionalMask += (1 << 20); // > 100 kHZ (if masknoise = 1 kHz)
        }
      }

      //Fill noisy in diagnostic
      mDiagnosticFrequency.fillNoisy(i + additionalMask, mReadoutWindowData.size());
    }
  }
}
