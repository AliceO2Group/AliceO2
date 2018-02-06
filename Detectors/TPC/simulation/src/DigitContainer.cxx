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

using namespace o2::TPC;

void DigitContainer::addDigit(size_t eventID, size_t trackID, const CRU& cru, TimeBin timeBin,
                              GlobalPadNumber globalPad, float signal)
{
  mEffectiveTimeBin = timeBin - mFirstTimeBin;
  if (mEffectiveTimeBin < 0.) {
    LOG(FATAL) << "TPC DigitCRU buffer misaligned ";
    LOG(DEBUG) << "for hit " << trackID << " CRU " << cru << " TimeBin " << timeBin << " First TimeBin "
               << mFirstTimeBin << " Global pad " << globalPad;
    LOG(FATAL) << FairLogger::endl;
    return;
  }
  if (cru.sector() != mSector) {
    LOG(FATAL) << "Digit for wrong sector " << cru.sector() << " added in sector " << mSector << FairLogger::endl;
  }
  /// If time bin outside specified range, the range of the vector is extended by one full drift time.
  while (mTimeBins.size() <= mEffectiveTimeBin) {
    mTimeBins.resize(mTimeBins.size() + 500);
  }
  mTimeBins[mEffectiveTimeBin].addDigit(eventID, trackID, cru, globalPad, signal);
}

void DigitContainer::fillOutputContainer(std::vector<Digit>* output,
                                         dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                                         std::vector<DigitMCMetaData>* debug, TimeBin eventTime, bool isContinuous)
{
  int nProcessedTimeBins = 0;
  TimeBin timeBin = mFirstTimeBin;
  for (auto& time : mTimeBins) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    /// OR the readout is triggered (i.e. not continuous) and we can dump everything in any case
    if ((nProcessedTimeBins + mFirstTimeBin < eventTime) || !isContinuous) {
      ++nProcessedTimeBins;
      time.fillOutputContainer(output, mcTruth, debug, mSector, timeBin);
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
  if (!isContinuous) {
    mFirstTimeBin = 0;
  }
}
