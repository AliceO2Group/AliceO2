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
/// \brief
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
  mTraits = traits;
}

float Vertexer::clustersToVertices(ROframe& event, const bool useMc, std::ostream& timeBenchmarkOutputStream)
{
  ROframe* eventptr = &event;
  float total{0.f};
  total += evaluateTask(&Vertexer::initialiseVertexer, "Vertexer initialisation", timeBenchmarkOutputStream, eventptr);
  total += evaluateTask(&Vertexer::findTracklets, "Tracklet finding", timeBenchmarkOutputStream);
#ifdef _ALLOW_DEBUG_TREES_ITS_
  if (useMc) {
    total += evaluateTask(&Vertexer::filterMCTracklets, "MC tracklets filtering", timeBenchmarkOutputStream);
  }
#endif
  total += evaluateTask(&Vertexer::validateTracklets, "Adjacent tracklets validation", timeBenchmarkOutputStream);
  total += evaluateTask(&Vertexer::findVertices, "Vertex finding", timeBenchmarkOutputStream);

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
} // namespace its
} // namespace o2
