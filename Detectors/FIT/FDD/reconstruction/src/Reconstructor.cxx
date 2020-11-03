// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Reconstructor.cxx
/// \brief Implementation of the FDD reconstruction

#include "CommonDataFormat/InteractionRecord.h"
#include "FDDReconstruction/Reconstructor.h"
#include "FairLogger.h"

using namespace o2::fdd;

//_____________________________________________________________________
void Reconstructor::process(const o2::fdd::Digit& digitBC, gsl::span<const o2::fdd::ChannelData> digitCh, std::vector<o2::fdd::RecPoint>& recPoints) const
{
  //Compute charge weighted average time
  Double_t timeFDA = 0, timeFDC = 0;
  Double_t weightFDA = 0.0, weightFDC = 0.0;

  for (const auto& channel : digitCh) {
    Float_t adc = channel.mChargeADC;
    Float_t time = channel.mTime;
    //LOG(INFO) <<adc <<"  "<<time;
    if (time == o2::InteractionRecord::DummyTime) {
      continue;
    }
    Float_t timeErr = 1;
    if (adc > 1) {
      timeErr = 1 / adc;
    }
    if (channel.mPMNumber < 8) {
      timeFDC += time / (timeErr * timeErr);
      weightFDC += 1. / (timeErr * timeErr);
    } else {
      timeFDA += time / (timeErr * timeErr);
      weightFDA += 1. / (timeErr * timeErr);
    }
  }
  timeFDA = (weightFDA > 1) ? timeFDA / weightFDA : o2::InteractionRecord::DummyTime;
  timeFDC = (weightFDC > 1) ? timeFDC / weightFDC : o2::InteractionRecord::DummyTime;

  recPoints.emplace_back(timeFDA, timeFDC, digitBC.getIntRecord());
}
//________________________________________________________
void Reconstructor::finish()
{
  // finalize digitization, if needed, flash remaining digits
  // if (!mContinuous)   return;
}
