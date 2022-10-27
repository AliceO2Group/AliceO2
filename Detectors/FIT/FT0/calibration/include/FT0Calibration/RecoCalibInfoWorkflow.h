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

/// \file   RecoCalibInfoWorkflow.h
///\ brief  Collect data for global offsets calibration
/// \author Alla.Maevskaya@cern.ch

#ifndef O2_RECOCALIBINFO_WORKFLOW
#define O2_RECOCALIBINFO_WORKFLOW

#include <fairlogger/Logger.h>
#include <Framework/ConfigContext.h>
#include <TMath.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsFT0/RecoCalibInfoObject.h"
#include "TStopwatch.h"
#include <string>
#include <vector>

using namespace o2::framework;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;

namespace o2::ft0
{

class RecoCalibInfoWorkflow final : public o2::framework::Task
{
 public:
  RecoCalibInfoWorkflow() {}
  void run(o2::framework::ProcessingContext& pc) final;
  void init(InitContext& ic) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  const float cSpeed = 0.029979246f; // speed of light in TOF units
  GID::mask_t mInputSources;
  TStopwatch mTimer;
};
framework::DataProcessorSpec getRecoCalibInfoWorkflow(bool useMC);

} // namespace o2::ft0

#endif /* O2_RECOCALIBINFO_WORKFLOW */
