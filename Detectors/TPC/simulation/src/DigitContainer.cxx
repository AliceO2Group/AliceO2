// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitContainer.cxx
/// \brief Implementation of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitContainer.h"
#include "FairLogger.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/ParameterElectronics.h"

using namespace o2::TPC;

void DigitContainer::addDigit(const MCCompLabel& label, const CRU& cru, TimeBin timeBin, GlobalPadNumber globalPad,
                              float signal)
{
  mEffectiveTimeBin = timeBin - mFirstTimeBin;
  if (mEffectiveTimeBin < 0.) {
    LOG(FATAL) << "TPC DigitTime buffer misaligned "
               << "for hit " << label.getTrackID() << " CRU " << cru << " TimeBin " << timeBin << " First TimeBin "
               << mFirstTimeBin << " Global pad " << globalPad;
    return;
  }
  /// If time bin outside specified range, the range of the vector is extended by one full drift time.
  while (mTimeBins.size() <= mEffectiveTimeBin) {
    mTimeBins.resize(mTimeBins.size() + 500);
  }
  mTimeBins[mEffectiveTimeBin].addDigit(label, cru, globalPad, signal);
}

void DigitContainer::fillOutputContainer(std::vector<Digit>& output,
                                         dataformats::MCTruthContainer<MCCompLabel>& mcTruth, const Sector& sector, TimeBin eventTime, bool isContinuous, bool finalFlush)
{
  int nProcessedTimeBins = 0;
  TimeBin timeBin = (isContinuous) ? mFirstTimeBin : 0;
  for (auto& time : mTimeBins) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    /// OR the readout is triggered (i.e. not continuous) and we can dump everything in any case, as long it is within one drift time interval
    if ((nProcessedTimeBins + mFirstTimeBin < eventTime) || !isContinuous || finalFlush) {
      if (!isContinuous && timeBin > mTmaxTriggered) {
        continue;
      }
      ++nProcessedTimeBins;
      auto& cdb = CDBInterface::instance();
      auto& eleParam = cdb.getParameterElectronics();
      const auto digitizationMode = eleParam.getDigitizationMode();

      switch (digitizationMode) {
        case DigitzationMode::FullMode: {
          time.fillOutputContainer<DigitzationMode::FullMode>(output, mcTruth, sector, timeBin);
          break;
        }
        case DigitzationMode::SubtractPedestal: {
          time.fillOutputContainer<DigitzationMode::SubtractPedestal>(output, mcTruth, sector, timeBin);
          break;
        }
        case DigitzationMode::NoSaturation: {
          time.fillOutputContainer<DigitzationMode::NoSaturation>(output, mcTruth, sector, timeBin);
          break;
        }
        case DigitzationMode::PropagateADC: {
          time.fillOutputContainer<DigitzationMode::PropagateADC>(output, mcTruth, sector, timeBin);
          break;
        }
      }
    } else {
      break;
    }
    timeBin++;
  }
  if (nProcessedTimeBins > 0) {
    mFirstTimeBin += nProcessedTimeBins;
    while (nProcessedTimeBins--) {
      mTimeBins.pop_front();
    }
  }
}
