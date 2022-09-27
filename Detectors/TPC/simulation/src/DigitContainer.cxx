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

/// \file DigitContainer.cxx
/// \brief Implementation of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitContainer.h"
#include <memory>
#include "FairLogger.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/ParameterElectronics.h"

using namespace o2::tpc;

void DigitContainer::fillOutputContainer(std::vector<Digit>& output,
                                         dataformats::MCTruthContainer<MCCompLabel>& mcTruth, std::vector<CommonMode>& commonModeOutput, const Sector& sector, TimeBin eventTimeBin, bool isContinuous, bool finalFlush)
{
  using Streamer = o2::utils::DebugStreamer;
  Streamer* debugStream = nullptr;
  if (Streamer::checkStream(o2::utils::StreamFlags::streamDigitFolding) || Streamer::checkStream(o2::utils::StreamFlags::streamDigits)) {
    mStreamer.setStreamer("debug_digits", "UPDATE");
    debugStream = &mStreamer;
  }

  const auto& eleParam = ParameterElectronics::Instance();
  const auto digitizationMode = eleParam.DigiMode;
  int nProcessedTimeBins = 0;
  TimeBin timeBin = (isContinuous) ? mFirstTimeBin : 0;

  const bool needsPrevDigArray = eleParam.doIonTail || eleParam.doSaturationTail;
  const bool needsEmptyTimeBins = needsPrevDigArray || eleParam.doNoiseEmptyPads;

  // TODO: make creation conditional
  if (needsPrevDigArray && !mPrevDigArr) {
    mPrevDigArr = std::make_unique<DigitTime::PrevDigitInfoArray>();
  }

  for (auto& time : mTimeBins) {
    /// the time bins between the last event and the timing of this event are uncorrelated and can be written out
    /// OR the readout is triggered (i.e. not continuous) and we can dump everything in any case, as long it is within one drift time interval
    if (!((nProcessedTimeBins + mFirstTimeBin < eventTimeBin) || !isContinuous || finalFlush)) {
      break;
    }

    if (!isContinuous && timeBin > mTmaxTriggered) {
      continue;
    }

    // fill also time bins without signal to get noise, ion tail and saturated signals
    if (needsEmptyTimeBins && !time) {
      time = new DigitTime;
    }

    // fmt::print("Processing secotor: {}, time bin: {}, mFirstTimeBin: {}, dgitTime: {}\n", sector.getSector(), timeBin, mFirstTimeBin, (void*)time);

    if (time) {
      switch (digitizationMode) {
        case DigitzationMode::FullMode: {
          time->fillOutputContainer<DigitzationMode::FullMode>(output, mcTruth, commonModeOutput, sector, timeBin, mPrevDigArr.get(), debugStream);
          break;
        }
        case DigitzationMode::ZeroSuppression: {
          time->fillOutputContainer<DigitzationMode::ZeroSuppression>(output, mcTruth, commonModeOutput, sector, timeBin, mPrevDigArr.get(), debugStream);
          break;
        }
        case DigitzationMode::SubtractPedestal: {
          time->fillOutputContainer<DigitzationMode::SubtractPedestal>(output, mcTruth, commonModeOutput, sector, timeBin, mPrevDigArr.get(), debugStream);
          break;
        }
        case DigitzationMode::NoSaturation: {
          time->fillOutputContainer<DigitzationMode::NoSaturation>(output, mcTruth, commonModeOutput, sector, timeBin, mPrevDigArr.get(), debugStream);
          break;
        }
        case DigitzationMode::PropagateADC: {
          time->fillOutputContainer<DigitzationMode::PropagateADC>(output, mcTruth, commonModeOutput, sector, timeBin, mPrevDigArr.get(), debugStream);
          break;
        }
      }
    }

    ++nProcessedTimeBins;
    ++timeBin;
  }

  if (nProcessedTimeBins > 0) {
    mFirstTimeBin += nProcessedTimeBins;
    while (nProcessedTimeBins--) {
      auto popped = mTimeBins.front();
      mTimeBins.pop_front();
      delete popped;
    }
  }
}
