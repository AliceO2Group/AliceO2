// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitCRU.cxx
/// \brief Implementation of the CRU container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/DigitTime.h"
#include "TPCBase/Mapper.h"

#include "FairLogger.h"

using namespace o2::TPC;

void DigitCRU::setDigit(size_t hitID, int timeBin, int row, int pad, float charge)
{
  mEffectiveTimeBin = timeBin - mFirstTimeBin;
  if(mEffectiveTimeBin < 0.) {
    LOG(FATAL) << "TPC DigitCRU buffer misaligned ";
    LOG(DEBUG) << "for hit " << hitID << " CRU " <<mCRU << " TimeBin " << timeBin << " First TimeBin " << mFirstTimeBin << " Row " << row << " Pad " << pad;
    LOG(FATAL) << FairLogger::endl;
    return;
  }
  /// If time bin outside specified range, the range of the vector is extended by one full drift time.
  while(getSize() <= mEffectiveTimeBin) {
    mTimeBins.resize(getSize() + mNTimeBins);
  }
  /// Check whether the container at this spot already contains an entry
  DigitTime *result = mTimeBins[mEffectiveTimeBin].get();
  if(result != nullptr) {
    mTimeBins[mEffectiveTimeBin]->setDigit(hitID, mCRU, row, pad, charge);
  }
  else {
    const Mapper& mapper = Mapper::instance();
    mTimeBins[mEffectiveTimeBin] = std::make_unique<DigitTime> (timeBin, mapper.getPadRegionInfo(CRU(mCRU).region()).getNumberOfPadRows());
    mTimeBins[mEffectiveTimeBin]->setDigit(hitID, mCRU, row, pad, charge);
  }
}

void DigitCRU::fillOutputContainer(std::vector<o2::TPC::Digit> *output, o2::dataformats::MCTruthContainer<o2::MCCompLabel> &mcTruth,
                                   std::vector<o2::TPC::DigitMCMetaData> *debug, int cru, int eventTime, bool isContinuous)
{
  int nProcessedTimeBins = 0;
  for(auto &aTime : mTimeBins) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    /// OR the readout is triggered (i.e. not continuous) and we can dump everything in any case
    if( ( nProcessedTimeBins + mFirstTimeBin < eventTime ) || !isContinuous) {
      ++nProcessedTimeBins;
      if(aTime == nullptr) continue;
      aTime->fillOutputContainer(output, mcTruth, debug, cru, aTime->getTimeBin(), mCommonModeContainer.getCommonMode(cru, aTime->getTimeBin()));
    }
    else break;
  }
  if(nProcessedTimeBins > 0) {
    mFirstTimeBin += nProcessedTimeBins;
    while(nProcessedTimeBins--) {
      mTimeBins.pop_front();
    }
  }
  if(!isContinuous) mFirstTimeBin = 0;
}
