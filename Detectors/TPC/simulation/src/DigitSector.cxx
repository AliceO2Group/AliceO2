// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitSector.cxx
/// \brief Implementation of the Sector container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitSector.h"

#include "FairLogger.h"

using namespace o2::TPC;

void DigitSector::setDigit(size_t eventID, size_t hitID, const CRU &cru, TimeBin timeBin, GlobalPadNumber globalPad, float charge)
{
  mEffectiveTimeBin = timeBin - mFirstTimeBin;
  if(mEffectiveTimeBin < 0.) {
    LOG(FATAL) << "TPC DigitSector buffer misaligned ";
    LOG(DEBUG) << "for event " << eventID << " hit " << hitID << " CRU " <<cru << " TimeBin " << timeBin << " First TimeBin " << mFirstTimeBin << " Global Pad " << globalPad;
    LOG(FATAL) << FairLogger::endl;
    return;
  }
  /// If time bin outside specified range, the range of the vector is extended by one full drift time.
  while(getSize() <= mEffectiveTimeBin) {
    mTimeBins.resize(getSize() + mNTimeBins);
  }
  mTimeBins[mEffectiveTimeBin].setDigit(eventID, hitID, cru, globalPad, charge);
}

void DigitSector::fillOutputContainer(std::vector<Digit> *output, dataformats::MCTruthContainer<MCCompLabel> &mcTruth,
                                      std::vector<DigitMCMetaData> *debug, Sector sector, TimeBin eventTime, bool isContinuous)
{
  TimeBin time = mFirstTimeBin;
  for(auto &aTime : mTimeBins) {
      /// GET TIME BIN INFORMATION
      aTime.fillOutputContainer(output, mcTruth, debug, sector, time++);
    }
  }

