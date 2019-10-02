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

using namespace o2::tpc;

void DigitContainer::fillOutputContainer(std::vector<Digit>& output,
                                         dataformats::MCTruthContainer<MCCompLabel>& mcTruth, std::vector<CommonMode>& commonModeOutput, const Sector& sector, TimeBin eventTimeBin, bool isContinuous, bool finalFlush)
{
  auto& eleParam = ParameterElectronics::Instance();
  const auto digitizationMode = eleParam.DigiMode;
  int nProcessedTimeBins = 0;
  TimeBin timeBin = (isContinuous) ? mFirstTimeBin : 0;
  for (auto& time : mTimeBins) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    /// OR the readout is triggered (i.e. not continuous) and we can dump everything in any case, as long it is within one drift time interval
    if ((nProcessedTimeBins + mFirstTimeBin < eventTimeBin) || !isContinuous || finalFlush) {
      if (!isContinuous && timeBin > mTmaxTriggered) {
        continue;
      }
      ++nProcessedTimeBins;

      switch (digitizationMode) {
        case DigitzationMode::FullMode: {
          time.fillOutputContainer<DigitzationMode::FullMode>(output, mcTruth, commonModeOutput, sector, timeBin);
          break;
        }
        case DigitzationMode::SubtractPedestal: {
          time.fillOutputContainer<DigitzationMode::SubtractPedestal>(output, mcTruth, commonModeOutput, sector, timeBin);
          break;
        }
        case DigitzationMode::NoSaturation: {
          time.fillOutputContainer<DigitzationMode::NoSaturation>(output, mcTruth, commonModeOutput, sector, timeBin);
          break;
        }
        case DigitzationMode::PropagateADC: {
          time.fillOutputContainer<DigitzationMode::PropagateADC>(output, mcTruth, commonModeOutput, sector, timeBin);
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
