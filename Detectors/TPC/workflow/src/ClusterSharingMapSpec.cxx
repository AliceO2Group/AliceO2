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

/// @file   ClusterSharingMapSpec.cxx
/// @brief Device to produce TPC clusters sharing map
/// \author ruben.shahoyan@cern.ch

#include <gsl/span>
#include <TStopwatch.h>
#include <vector>
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "GPUO2InterfaceRefit.h"
#include "GPUO2InterfaceUtils.h"
#include "TPCWorkflow/ClusterSharingMapSpec.h"
#include "DataFormatsParameters/GRPECSObject.h"

using namespace o2::framework;
using namespace o2::tpc;

void ClusterSharingMapSpec::run(ProcessingContext& pc)
{
  TStopwatch timer;
  static int nHBPerTF = 0;
  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  const auto tracksTPCClRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
  const auto& clustersTPC = getWorkflowTPCInput(pc);
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) { // new run is starting
    auto grp = pc.inputs().get<o2::parameters::GRPECSObject*>("grpecs");
    nHBPerTF = grp->getNHBFPerTF();
    LOGP(info, "Will use {} HB per TF from GRPECS", nHBPerTF);
  }

  std::shared_ptr<o2::gpu::GPUParam> param = o2::gpu::GPUO2InterfaceUtils::getFullParamShared(0.f, nHBPerTF);
  auto& bufVecSh = pc.outputs().make<std::vector<unsigned char>>(Output{o2::header::gDataOriginTPC, "CLSHAREDMAP", 0}, clustersTPC->clusterIndex.nClustersTotal);
  size_t occupancyMapSizeBytes = o2::gpu::GPUO2InterfaceRefit::fillOccupancyMapGetSize(nHBPerTF, param.get());
  auto& bufVecOcc = pc.outputs().make<std::vector<unsigned int>>(Output{o2::header::gDataOriginTPC, "TPCOCCUPANCYMAP", 0}, occupancyMapSizeBytes / sizeof(int));
  o2::gpu::GPUO2InterfaceRefit::fillSharedClustersAndOccupancyMap(&clustersTPC->clusterIndex, tracksTPC, tracksTPCClRefs.data(), bufVecSh.data(), bufVecOcc.data(), nHBPerTF, param.get());

  timer.Stop();
  LOGF(info, "Timing for TPC clusters sharing map creation: Cpu: %.3e Real: %.3e s", timer.CpuTime(), timer.RealTime());
}
