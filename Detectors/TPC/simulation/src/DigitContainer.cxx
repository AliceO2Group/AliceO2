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
#include <iostream>

using namespace o2::TPC;

void DigitContainer::setUp(const short sector, const TimeBin timeBinEvent) {
  mSectorID = sector;
  mSector[getBufferPosition(sector)].init(sector, timeBinEvent);
  mSector[getBufferPosition(getSectorRight(sector))].init(getSectorRight(sector), timeBinEvent);
  mSector[getBufferPosition(getSectorLeft(sector))].init(getSectorLeft(sector), timeBinEvent);
}

unsigned short DigitContainer::getSectorLeft(const short sector) const
 {
   const int modulus = sector%18;
   int offsetSector = static_cast<int>(sector/18) * 18;
   if(modulus == 0) offsetSector += 18;
   return offsetSector+modulus-1;
 }

unsigned short DigitContainer::getSectorRight(const short sector) const
 {
   const int modulus = sector%18;
   int offsetSector = static_cast<int>(sector/18) * 18;
   if(modulus == 17) offsetSector -= 18;
   return offsetSector+modulus+1;
 }

bool DigitContainer::checkNeighboursProcessed(const short sector) const
{
  return (mSectorProcessed[sector] && mSectorProcessed[getSectorRight(sector)] && mSectorProcessed[getSectorLeft(sector)]);
}

unsigned short DigitContainer::getBufferPosition(const short sector)
{
  if(mSectorMapping[sector] > -1) return mSectorMapping[sector];
  else {
    mSectorMapping[sector] = mNextFreePosition;
    return mNextFreePosition++;
  }
}

void DigitContainer::addDigit(size_t eventID, size_t hitID, const CRU &cru, TimeBin timeBin, GlobalPadNumber globalPad, float charge)
{
  const int sector = cru.sector();
  mSector[getBufferPosition(sector)].setDigit(eventID, hitID, cru, timeBin, globalPad, charge);
}


void DigitContainer::fillOutputContainer(std::vector<Digit> *output, dataformats::MCTruthContainer<MCCompLabel> &mcTruth,
                                         std::vector<DigitMCMetaData> *debug, TimeBin eventTime, bool isContinuous, bool isFinal)
{
  mSectorProcessed[mSectorID] = true;
  for(int s=0; s<4; ++s) {
    if(!checkNeighboursProcessed(s)) continue;
    std::cout << "writing sector " << s << "\n";
    /// \todo Use CRU here instead of the sector!
    mSector[s].fillOutputContainer(output, mcTruth, debug, mSector[s].getSector(), eventTime, isContinuous);
    mSectorMapping[mSector[s].getSector()] = -1;
    mNextFreePosition = (s < mNextFreePosition) ? s : mNextFreePosition;
  }
}



