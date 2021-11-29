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

/// @file   BarrelAlignmentSpec.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "AlignmentWorkflow/BarrelAlignmentSpec.h"

#include "Align/Controller.h"

#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"

#include "Headers/DataHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"

/*
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/DPLAlpideParam.h"
*/

using namespace o2::framework;
using namespace o2::globaltracking;
using namespace o2::align;

using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace align
{

class BarrelAlignmentSpec : public Task
{
 public:
  BarrelAlignmentSpec(std::shared_ptr<DataRequest> dr, DetID::mask_t m) : mDataRequest(dr), mDetMask{m} {}
  ~BarrelAlignmentSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  void updateTimeDependentParams();

  DetID::mask_t mDetMask{};
  std::unique_ptr<Controller> mController;
  std::shared_ptr<DataRequest> mDataRequest;
  TStopwatch mTimer;
};

void BarrelAlignmentSpec::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  mController = std::make_unique<Controller>(mDetMask);
}

void BarrelAlignmentSpec::updateTimeDependentParams()
{
  //
}

void BarrelAlignmentSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  mController->process(recoData);

  mTimer.Stop();
}

void BarrelAlignmentSpec::endOfStream(EndOfStreamContext& ec)
{
  //mBarrelAlign.end();
  LOGF(info, "Barrel alignment data pereparation total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getBarrelAlignmentSpec(GTrackID::mask_t src, DetID::mask_t dets)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, false);
  dataRequest->requestClusters(src, false);

  return DataProcessorSpec{
    "barrel-alignment",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<BarrelAlignmentSpec>(dataRequest, dets)},
    Options{
      {"its-dictionary-path", VariantType::String, "", {"Path of the cluster-topology dictionary file"}},
      {"material-lut-path", VariantType::String, "", {"Path of the material LUT file"}}}};
}

} // namespace align
} // namespace o2
