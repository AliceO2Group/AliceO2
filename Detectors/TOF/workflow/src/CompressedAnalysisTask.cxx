// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedAnalysisTask.cxx
/// @author Roberto Preghenella
/// @since  2020-09-04
/// @brief  TOF compressed data analysis task

#include "TOFWorkflow/CompressedAnalysisTask.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include <fairmq/FairMQDevice.h>

using namespace o2::framework;

namespace o2
{
namespace tof
{

void CompressedAnalysisTask::init(InitContext& ic)
{

  auto conetmode = ic.options().get<bool>("tof-compressed-analysis-conet-mode");
  auto filename = ic.options().get<std::string>("tof-compressed-analysis-filename");
  auto function = ic.options().get<std::string>("tof-compressed-analysis-function");

  if (filename.empty()) {
    LOG(ERROR) << "No analysis filename defined";
    mStatus = true;
    return;
  }

  if (function.empty()) {
    LOG(ERROR) << "No analysis function defined";
    mStatus = true;
    return;
  }

  mAnalysis = GetFromMacro<CompressedAnalysis*>(filename, function, "CompressedAnalysis*", "compressed_analysis");
  if (!mAnalysis) {
    LOG(ERROR) << "Could not retrieve analysis from file: " << filename;
    mStatus = true;
    return;
  }

  mAnalysis->setDecoderCONET(conetmode);
  mAnalysis->initialize();

  auto finishFunction = [this]() {
    LOG(INFO) << "CompressedBaseTask finish";
    mAnalysis->finalize();
  };
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

void CompressedAnalysisTask::run(ProcessingContext& pc)
{

  /** check status **/
  if (mStatus) {
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  /** loop over inputs routes **/
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid())
      continue;

    /** loop over input parts **/
    for (auto const& ref : iit) {

      const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;

      mAnalysis->setDecoderBuffer(payloadIn);
      mAnalysis->setDecoderBufferSize(payloadInSize);
      mAnalysis->run();
    }
  }
}

} // namespace tof
} // namespace o2
