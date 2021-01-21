// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TPCWorkflow/ClusterSharingMapSpec.h"

using namespace o2::framework;
using namespace o2::tpc;

void ClusterSharingMapSpec::run(ProcessingContext& pc)
{
  TStopwatch timer;

  const auto tracksTPC = pc.inputs().get<gsl::span<o2::tpc::TrackTPC>>("trackTPC");
  const auto tracksTPCClRefs = pc.inputs().get<gsl::span<o2::tpc::TPCClRefElem>>("trackTPCClRefs");
  const auto& clustersTPC = getWorkflowTPCInput(pc);

  auto& bufVec = pc.outputs().make<std::vector<unsigned char>>(Output{o2::header::gDataOriginTPC, "CLSHAREDMAP", 0}, clustersTPC->clusterIndex.nClustersTotal);
  o2::gpu::GPUO2InterfaceRefit::fillSharedClustersMap(&clustersTPC->clusterIndex, tracksTPC, tracksTPCClRefs.data(), bufVec.data());

  timer.Stop();
  LOGF(INFO, "Timing for TPC clusters sharing map creation: Cpu: %.3e Real: %.3e s", timer.CpuTime(), timer.RealTime());
}
