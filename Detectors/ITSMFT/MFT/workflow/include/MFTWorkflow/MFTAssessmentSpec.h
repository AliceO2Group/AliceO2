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

#ifndef ALICEO2_MFT_ASSESSMENT_DEVICE_H
#define ALICEO2_MFT_ASSESSMENT_DEVICE_H

/// @file   MFTAssessmentSpec.h

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "MFTAssessment/MFTAssessment.h"
#include "TStopwatch.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{
class MFTAssessmentSpec : public Task
{
 public:
  MFTAssessmentSpec(bool useMC, bool processGen, bool finalizeAnalysis = false) : mUseMC(useMC),
                                                                                  mProcessGen(processGen),
                                                                                  mFinalizeAnalysis(finalizeAnalysis){};
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void sendOutput(DataAllocator& output);
  std::unique_ptr<o2::mft::MFTAssessment> mMFTAssessment;
  bool mUseMC = true;
  bool mProcessGen = false;
  bool mFinalizeAnalysis = false;
  enum TimerIDs { SWTot,
                  SWQCAsync,
                  SWTrackables,
                  SWGenerated,
                  SWRecoAndTrue,
                  SWAnalysis,
                  NStopWatches };
  static constexpr std::string_view TimerName[] = {"Total",
                                                   "ProcessAsync",
                                                   "ProcessTrackables",
                                                   "ProcessGenerated",
                                                   "ProcessRecoAndTrue",
                                                   "Analysis"};
  TStopwatch mTimer[NStopWatches];
};

DataProcessorSpec getMFTAssessmentSpec(bool useMC, bool processGen, bool finalizeAnalysis = false);

} // namespace mft
} // namespace o2

#endif