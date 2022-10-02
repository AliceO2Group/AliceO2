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

#include <fairlogger/Logger.h>
#include <Framework/ConfigContext.h>
#include <TMath.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsFT0/RecoCalibInfoObject.h"

using namespace o2::framework;

namespace o2::ft0
{

class RecoCalibInfoWorkflow final : public o2::framework::Task
{
  using DataRequest = o2::globaltracking::DataRequest;

 public:
  /* void collectBCs(gsl::span<const o2::ft0::RecPoints>& ft0RecPoints, */
  /*                 gsl::span<const o2::dataformats::PrimaryVertex>& primVertices, */
  /*                 std::map<uint64_t, int>& bcsMap); */
  void run(o2::framework::ProcessingContext& pc) final
  {

    o2::globaltracking::RecoContainer recoData;
    recoData.collectData(pc, *mDataRequest);
    LOG(info) << " @@@read RecoContainer";
    auto primVertices = recoData.getPrimaryVertices();
    LOG(info) << "@@@ primVertices  ";
    auto ft0RecPoints = recoData.getFT0RecPoints();
    LOG(info) << "@@@@ read T0 recpoints ";
    std::map<uint64_t, o2::dataformats::PrimaryVertex const*> bcsMap;
    for (auto& vertex : primVertices) {
      auto& timeStamp = vertex.getTimeStamp();
      double tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
      uint64_t globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
      auto [iter, inserted] = bcsMap.try_emplace(globalBC, &vertex);
      if (!inserted)
        iter->second = nullptr;
    }
    /* collectBCs(ft0RecPoints, primVertices, bcsMap); */

    for (auto& ft0RecPoint : ft0RecPoints) {
      uint64_t bc = ft0RecPoint.getInteractionRecord().toLong();
      /* uint64_t bc = globalBC; */
      auto item = bcsMap.find(bc);
      if (item == bcsMap.end() || item->second == nullptr) {
        LOG(fatal) << "Error: could not find a corresponding BC ID for a FT0 rec. point; BC = " << bc;
        continue;
      }
      auto& vertex = *item->second;
      /* int bcID = -1; */
      /* if (item != bcsMap.end()) { */
      /*   bcID = item->second; */
      auto currentVertex = vertex.getZ();
      ushort ncont = vertex.getNContributors();
      LOG(info) << "@@@ currentVertex " << currentVertex << " ncont " << int(ncont);
      if (ncont == 0)
        continue;
      auto shift = currentVertex / TMath::C();
      LOG(info) << " BC  t0 " << bc;
      short t0A = ft0RecPoint.getCollisionTimeA() + shift;
      short t0C = ft0RecPoint.getCollisionTimeC() - shift;
      short t0AC = ft0RecPoint.getCollisionTimeMean();

      /* auto recpoints = */
      /*     pc.inputs().get<gsl::span<o2::ft0::RecPoints>>("recpoints"); */
      auto& calib_data = pc.outputs().make<std::vector<o2::ft0::RecoCalibInfoObject>>(o2::framework::OutputRef{"calib", 0});
      calib_data.emplace_back(t0A, t0C, t0AC);
    }
  }

 private:
  std::shared_ptr<DataRequest> mDataRequest;
};

} // namespace o2::ft0

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const&)
{

  DataProcessorSpec dataProcessorSpec{
    "RecoCalibInfoWorkflow",
    Inputs{
      {{"recpoints"}, "FT0", "FT0CLUSTER"},
    },
    Outputs{
      {{"calib"}, "FT0", "CALIB_INFO"}},
    AlgorithmSpec{adaptFromTask<o2::ft0::RecoCalibInfoWorkflow>()},
    Options{}};

  WorkflowSpec workflow;
  workflow.emplace_back(dataProcessorSpec);
  return workflow;
}
