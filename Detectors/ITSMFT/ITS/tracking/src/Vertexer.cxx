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
  mVertParams.resize(1);
  mTraits = traits;
}

float Vertexer::clustersToVertices(std::function<void(std::string s)> logger)
{
  float total{0.f};
  TrackingParameters trkPars;
  TimeFrameGPUParameters tfGPUpar;
  mTraits->updateVertexingParameters(mVertParams, tfGPUpar);
  LOGP(info, "[1].phiCut {} [1].tanLambdaCut {}", mVertParams[1].phiCut, mVertParams[1].tanLambdaCut);
  for (int iteration = 0; iteration < std::min(mVertParams[0].nIterations, (int)mVertParams.size()); ++iteration) {
    logger(fmt::format("ITS Seeding vertexer iteration {} summary:", iteration));
    trkPars.PhiBins = mTraits->getVertexingParameters()[0].PhiBins;
    trkPars.ZBins = mTraits->getVertexingParameters()[0].ZBins;
    total += evaluateTask(&Vertexer::initialiseVertexer, "Vertexer initialisation", logger, trkPars, iteration);
    total += evaluateTask(&Vertexer::findTracklets, "Vertexer tracklet finding", logger, iteration);
    total += evaluateTask(&Vertexer::validateTracklets, "Vertexer adjacent tracklets validation", logger, iteration);
    total += evaluateTask(&Vertexer::findVertices, "Vertexer vertex finding", logger, iteration);
  }
  printEpilog(logger, total);
  return total;
}

float Vertexer::clustersToVerticesHybrid(std::function<void(std::string s)> logger)
{
  float total{0.f};
  TrackingParameters trkPars;
  trkPars.PhiBins = mTraits->getVertexingParameters()[0].PhiBins;
  trkPars.ZBins = mTraits->getVertexingParameters()[0].ZBins;
  total += evaluateTask(&Vertexer::initialiseVertexerHybrid, "Hybrid Vertexer initialisation", logger, trkPars);
  total += evaluateTask(&Vertexer::findTrackletsHybrid, "Hybrid Vertexer tracklet finding", logger);
  total += evaluateTask(&Vertexer::validateTrackletsHybrid, "Hybrid Vertexer adjacent tracklets validation", logger);
  total += evaluateTask(&Vertexer::findVerticesHybrid, "Hybrid Vertexer vertex finding", logger);
  printEpilog(logger, total);
  return total;
}

void Vertexer::getGlobalConfiguration()
{
  auto& vc = o2::its::VertexerParamConfig::Instance();
  vc.printKeyValues(true, true);
  auto& grc = o2::its::GpuRecoParamConfig::Instance();

  // This is odd: we override only the parameters for the first iteration.
  // Variations for the next iterations are set in the trackingInterfrace.
  mVertParams[0].nIterations = vc.nIterations;
  mVertParams[0].allowSingleContribClusters = vc.allowSingleContribClusters;
  mVertParams[0].zCut = vc.zCut;
  mVertParams[0].phiCut = vc.phiCut;
  mVertParams[0].pairCut = vc.pairCut;
  mVertParams[0].clusterCut = vc.clusterCut;
  mVertParams[0].histPairCut = vc.histPairCut;
  mVertParams[0].tanLambdaCut = vc.tanLambdaCut;
  mVertParams[0].lowMultBeamDistCut = vc.lowMultBeamDistCut;
  mVertParams[0].vertNsigmaCut = vc.vertNsigmaCut;
  mVertParams[0].vertRadiusSigma = vc.vertRadiusSigma;
  mVertParams[0].trackletSigma = vc.trackletSigma;
  mVertParams[0].maxZPositionAllowed = vc.maxZPositionAllowed;
  mVertParams[0].clusterContributorsCut = vc.clusterContributorsCut;
  mVertParams[0].maxTrackletsPerCluster = vc.maxTrackletsPerCluster;
  mVertParams[0].phiSpan = vc.phiSpan;
  mVertParams[0].nThreads = vc.nThreads;
  mVertParams[0].ZBins = vc.ZBins;
  mVertParams[0].PhiBins = vc.PhiBins;
}

void Vertexer::adoptTimeFrame(TimeFrame& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

void Vertexer::printEpilog(std::function<void(std::string s)> logger, const float total)
{
  logger(fmt::format(" - Timeframe {} vertexing completed in: {} ms, using {} thread(s).", mTimeFrameCounter++, total, mTraits->getNThreads()));
}

} // namespace its
} // namespace o2
