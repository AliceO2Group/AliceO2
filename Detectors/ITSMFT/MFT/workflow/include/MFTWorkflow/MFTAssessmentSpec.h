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

using namespace o2::framework;

namespace o2
{
namespace mft
{
class MFTAssessmentSpec : public Task
{
 public:
  MFTAssessmentSpec(bool useMC) : mUseMC(useMC){};
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void sendOutput(DataAllocator& output);
  std::unique_ptr<o2::mft::MFTAssessment> mMFTAssessment;
  bool mUseMC = true;
};

DataProcessorSpec getMFTAssessmentSpec(bool useMC);

} // namespace mft
} // namespace o2

#endif