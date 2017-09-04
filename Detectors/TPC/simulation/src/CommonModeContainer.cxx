// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CommonModeContainer.cxx
/// \brief Implementation of the Common Mode computation
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/CommonModeContainer.h"

#include "FairLogger.h"

using namespace o2::TPC;

void CommonModeContainer::addDigit(const CRU cru, const int timeBin, const float signal)
{
  mEffectiveTimeBin = timeBin-mFirstTimeBin;
  if(mEffectiveTimeBin < 0.) {
    LOG(FATAL) << "TPC CommonMode buffer misaligned for CRU " << cru << " TimeBin " << timeBin << " First TimeBin " << mFirstTimeBin << FairLogger::endl;
    return;
  }
  /// If time bin outside specified range, the range of the vector is extended by one full drift time.
  while(static_cast<int>(mCommonModeContainer.size()) <= mEffectiveTimeBin) {
    mCommonModeContainer.resize(static_cast<int>(mCommonModeContainer.size()) + 500);
  }

  const int sector = cru.sector();
  const int gemStack = static_cast<int>(cru.gemStack());
  mCommonModeContainer[mEffectiveTimeBin][4*sector+gemStack]+=signal;
}

void CommonModeContainer::cleanUp(int eventTime, bool isContinuous)
{
  int nProcessedTimeBins = 0;
  for(auto &commonMode : mCommonModeContainer) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    /// OR the readout is triggered (i.e. not continuous) and we can dump everything in any case
    if( ( nProcessedTimeBins + mFirstTimeBin < eventTime ) || !isContinuous) {
      ++nProcessedTimeBins;
    }
    else break;
  }
  if(nProcessedTimeBins > 0) {
    mFirstTimeBin += nProcessedTimeBins;
    while(nProcessedTimeBins--) {
      mCommonModeContainer.pop_front();
    }
  }
  if(!isContinuous) {
    mFirstTimeBin = 0;
    mCommonModeContainer.resize(500);
  }
}
