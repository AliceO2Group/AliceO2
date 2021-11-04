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
///
/// \file Vertexer.cxx
/// \author Matteo Concas mconcas@cern.ch
///

#include "ITStracking/Vertexer.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/TrackingConfigParam.h"

#include <array>

namespace o2
{
namespace its
{

Vertexer::Vertexer(VertexerTraits* traits)
{
  if (!traits) {
    LOG(fatal) << "nullptr passed to ITS vertexer construction.";
  }
  mTraits = traits;
}

float Vertexer::clustersToVertices(const bool useMc, std::function<void(std::string s)> logger)
{
  float total{0.f};
  TrackingParameters trkPars;
  MemoryParameters memPars;
  total += evaluateTask(&Vertexer::initialiseVertexer, true, "Vertexer initialisation", logger, memPars, trkPars);
  total += evaluateTask(&Vertexer::findTracklets, true, "Tracklet finding", logger);
  // #ifdef _ALLOW_DEBUG_TREES_ITS_
  //   if (useMc) {
  //     total += evaluateTask(&Vertexer::filterMCTracklets, "MC tracklets filtering", logger);
  //   }
  // #endif
  total += evaluateTask(&Vertexer::validateTracklets, false, "Adjacent tracklets validation", logger);
  // total += evaluateTask(&Vertexer::findVertices, false, "Vertex finding", logger);

  return total;
}

void Vertexer::findVertices()
{
  mTraits->computeVertices();
}

void Vertexer::findHistVertices()
{
  mTraits->computeHistVertices();
}

void Vertexer::getGlobalConfiguration()
{
  auto& vc = o2::its::VertexerParamConfig::Instance();

  VertexingParameters verPar;
  verPar.zCut = vc.zCut;
  verPar.phiCut = vc.phiCut;
  verPar.pairCut = vc.pairCut;
  verPar.clusterCut = vc.clusterCut;
  verPar.histPairCut = vc.histPairCut;
  verPar.tanLambdaCut = vc.tanLambdaCut;
  verPar.clusterContributorsCut = vc.clusterContributorsCut;
  verPar.phiSpan = vc.phiSpan;

  mTraits->updateVertexingParameters(verPar);
}

void Vertexer::adoptTimeFrame(TimeFrame& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}
} // namespace its
} // namespace o2
