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

/// \file   RecoCalibInfoWorkflow.cxx
///\ brief  Collect data for global offsets calibration
/// \author Alla.Maevskaya@cern.ch

#include <fairlogger/Logger.h>
#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsFT0/RecPoints.h"
#include "FT0Calibration/RecoCalibInfoWorkflow.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DataFormatsFT0/RecoCalibInfoObject.h"
#include <TMath.h>
#include <vector>
#include <map>

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;

namespace o2::ft0
{
void RecoCalibInfoWorkflow::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
}
void RecoCalibInfoWorkflow::run(o2::framework::ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  auto primVertices = recoData.getPrimaryVertices();
  auto ft0RecPoints = recoData.getFT0RecPoints();
  std::map<uint64_t, o2::dataformats::PrimaryVertex const*> bcsMap;
  auto& calib_data = pc.outputs().make<std::vector<o2::ft0::RecoCalibInfoObject>>(o2::framework::OutputRef{"calib", 0});
  calib_data.reserve(ft0RecPoints.size());
  for (auto& vertex : primVertices) {
    auto& timeStamp = vertex.getTimeStamp();
    double tsTimeStamp = timeStamp.getTimeStamp() * 1E3; // mus to ns
    uint64_t globalBC = std::round(tsTimeStamp / o2::constants::lhc::LHCBunchSpacingNS);
    LOG(debug) << "PrimVertices " << globalBC;
    auto [iter, inserted] = bcsMap.try_emplace(globalBC, &vertex);
    if (!inserted) {
      iter->second = nullptr;
    }
  }
  for (auto& ft0RecPoint : ft0RecPoints) {
    uint64_t bc = ft0RecPoint.getInteractionRecord().toLong();
    auto item = bcsMap.find(bc);
    LOG(debug) << " <<ft0RecPoints " << bc;
    if (item == bcsMap.end() || item->second == nullptr) {
      LOG(debug) << "Error: could not find a corresponding BC ID for a FT0 rec. point; BC = " << bc;
      continue;
    }
    auto& vertex = *item->second;
    auto currentVertex = vertex.getZ();
    ushort ncont = vertex.getNContributors();
    LOG(debug) << "CurrentVertex " << currentVertex << " ncont " << int(ncont);
    if (ncont < 3) {
      continue;
    }
    auto shift = currentVertex / cSpeed;
    short t0A = ft0RecPoint.getCollisionTimeA() + shift;
    short t0C = ft0RecPoint.getCollisionTimeC() - shift;
    short t0AC = ft0RecPoint.getCollisionTimeMean();
    LOG(debug) << " BC  t0  " << bc << " shift " << shift << " A " << t0A << " C " << t0C << " AC " << t0AC;
    calib_data.emplace_back(t0A, t0C, t0AC);
  }
  mTimer.Stop();
}
void RecoCalibInfoWorkflow::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "Reco calib info workflow  dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getRecoCalibInfoWorkflow(GID::mask_t src, bool useMC)
{
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestPrimaryVertertices(false);
  dataRequest->requestFT0RecPoints(false);

  return DataProcessorSpec{
    "ft0-calib-reco",
    dataRequest->inputs,
    Outputs{
      {{"calib"}, "FT0", "CALIB_INFO"}},
    AlgorithmSpec{adaptFromTask<o2::ft0::RecoCalibInfoWorkflow>(src, dataRequest)},
    Options{}};
}

}; // namespace o2::ft0
