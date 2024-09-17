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
  TrackingParameters trkPars;
  TimeFrameGPUParameters tfGPUpar;
  mTraits->updateVertexingParameters(mVertParams, tfGPUpar);
  float timeTracklet{0.f}, timeSelection{0.f}, timeVertexing{0.f}, timeInit{0.f};
  for (int iteration = 0; iteration < std::min(mVertParams[0].nIterations, (int)mVertParams.size()); ++iteration) {
    unsigned int nTracklets01, nTracklets12;
    logger(fmt::format("ITS Seeding vertexer iteration {} summary:", iteration));
    trkPars.PhiBins = mTraits->getVertexingParameters()[0].PhiBins;
    trkPars.ZBins = mTraits->getVertexingParameters()[0].ZBins;
    auto timeInitIteration = evaluateTask(
      &Vertexer::initialiseVertexer, "Vertexer initialisation", [](std::string) {}, trkPars, iteration);
    auto timeTrackletIteration = evaluateTask(
      &Vertexer::findTracklets, "Vertexer tracklet finding", [](std::string) {}, iteration);
    nTracklets01 = mTimeFrame->getTotalTrackletsTF(0);
    nTracklets12 = mTimeFrame->getTotalTrackletsTF(1);
    auto timeSelectionIteration = evaluateTask(
      &Vertexer::validateTracklets, "Vertexer tracklets validation", [](std::string) {}, iteration);
    auto timeVertexingIteration = evaluateTask(
      &Vertexer::findVertices, "Vertexer vertex finding", [](std::string) {}, iteration);
    printEpilog(logger, false, nTracklets01, nTracklets12, mTimeFrame->getNLinesTotal(), mTimeFrame->getTotVertIteration()[iteration], timeInitIteration, timeTrackletIteration, timeSelectionIteration, timeVertexingIteration);
    timeInit += timeInitIteration;
    timeTracklet += timeTrackletIteration;
    timeSelection += timeSelectionIteration;
    timeVertexing += timeVertexingIteration;
  }

  return timeInit + timeTracklet + timeSelection + timeVertexing;
}

float Vertexer::clustersToVerticesHybrid(std::function<void(std::string s)> logger)
{
  TrackingParameters trkPars;
  TimeFrameGPUParameters tfGPUpar;
  float timeTracklet, timeSelection, timeVertexing, timeInit;
  mTraits->updateVertexingParameters(mVertParams, tfGPUpar);
  for (int iteration = 0; iteration < std::min(mVertParams[0].nIterations, (int)mVertParams.size()); ++iteration) {
    unsigned int nTracklets01, nTracklets12;
    logger(fmt::format("ITS Hybrid seeding vertexer iteration {} summary:", iteration));
    trkPars.PhiBins = mTraits->getVertexingParameters()[0].PhiBins;
    trkPars.ZBins = mTraits->getVertexingParameters()[0].ZBins;
    auto timeInitIteration = evaluateTask(
      &Vertexer::initialiseVertexerHybrid, "Hybrid Vertexer initialisation", [](std::string) {}, trkPars, iteration);
    auto timeTrackletIteration = evaluateTask(
      &Vertexer::findTrackletsHybrid, "Hybrid Vertexer tracklet finding", [](std::string) {}, iteration);
    nTracklets01 = mTimeFrame->getTotalTrackletsTF(0);
    nTracklets12 = mTimeFrame->getTotalTrackletsTF(1);
    auto timeSelectionIteration = evaluateTask(
      &Vertexer::validateTrackletsHybrid, "Hybrid Vertexer adjacent tracklets validation", [](std::string) {}, iteration);
    auto timeVertexingIteration = evaluateTask(
      &Vertexer::findVerticesHybrid, "Hybrid Vertexer vertex finding", [](std::string) {}, iteration);

    printEpilog(logger, true, nTracklets01, nTracklets12, mTimeFrame->getNLinesTotal(), mTimeFrame->getTotVertIteration().size(), timeInit, timeTracklet, timeSelection, timeVertexing);
    timeInit += timeInitIteration;
    timeTracklet += timeTrackletIteration;
    timeSelection += timeSelectionIteration;
    timeVertexing += timeVertexingIteration;
  }

  return timeInit + timeTracklet + timeSelection + timeVertexing;
}

void Vertexer::getGlobalConfiguration()
{
  auto& vc = o2::its::VertexerParamConfig::Instance();
  vc.printKeyValues(true, true);
  auto& grc = o2::its::GpuRecoParamConfig::Instance();

  // This is odd: we override only the parameters for the first iteration.
  // Variations for the next iterations are set in the trackingInterfrace.
  mVertParams[0].nIterations = vc.nIterations;
  mVertParams[0].deltaRof = vc.deltaRof;
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

void Vertexer::printEpilog(std::function<void(std::string s)> logger,
                           bool isHybrid,
                           const unsigned int trackletN01, const unsigned int trackletN12, const unsigned selectedN, const unsigned int vertexN,
                           const float initT, const float trackletT, const float selecT, const float vertexT)
{
  float total = initT + trackletT + selecT + vertexT;
  logger(fmt::format(" - {}Vertexer: found {} | {} tracklets in: {} ms", isHybrid ? "Hybrid" : "", trackletN01, trackletN12, trackletT));
  logger(fmt::format(" - {}Vertexer: selected {} tracklets in: {} ms", isHybrid ? "Hybrid" : "", selectedN, selecT));
  logger(fmt::format(" - {}Vertexer: found {} vertices in: {} ms", isHybrid ? "Hybrid" : "", vertexN, vertexT));
  // logger(fmt::format(" - Timeframe {} vertexing completed in: {} ms, using {} thread(s).", mTimeFrameCounter++, total, mTraits->getNThreads()));
}

} // namespace its
} // namespace o2
