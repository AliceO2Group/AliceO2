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

float Vertexer::clustersToVertices(std::function<void(std::string s)> logger)
{
  float timeTracklet, timeSelection, timeVertexing, timeInit;
  unsigned int nTracklets01, nTracklets12, nSelectedTracklets;
  TrackingParameters trkPars;
  trkPars.PhiBins = mTraits->getVertexingParameters().PhiBins;
  trkPars.ZBins = mTraits->getVertexingParameters().ZBins;
  timeInit += evaluateTask(&Vertexer::initialiseVertexer, "Vertexer initialisation", [](std::string) {}, trkPars);
  timeTracklet = evaluateTask(&Vertexer::findTracklets, "Vertexer tracklet finding", [](std::string) {});
  nTracklets01 = mTimeFrame->getTotalTrackletsTF(0);
  nTracklets12 = mTimeFrame->getTotalTrackletsTF(1);
  timeSelection = evaluateTask(&Vertexer::validateTracklets, "Vertexer adjacent tracklets validation", [](std::string) {});
  nSelectedTracklets = mTimeFrame->getNLinesTotal();
  timeVertexing = evaluateTask(&Vertexer::findVertices, "Vertexer vertex finding", [](std::string) {});
  printEpilog(logger, false, nTracklets01, nTracklets12, nSelectedTracklets, timeInit, timeTracklet, timeSelection, timeVertexing);
  return timeInit + timeTracklet + timeSelection + timeVertexing;
}

float Vertexer::clustersToVerticesHybrid(std::function<void(std::string s)> logger)
{
  float timeTracklet, timeSelection, timeVertexing, timeInit;
  unsigned int nTracklets01, nTracklets12, nSelectedTracklets;
  TrackingParameters trkPars;
  trkPars.PhiBins = mTraits->getVertexingParameters().PhiBins;
  trkPars.ZBins = mTraits->getVertexingParameters().ZBins;
  timeInit += evaluateTask(
    &Vertexer::initialiseVertexerHybrid, "Hybrid Vertexer initialisation", [](std::string) {}, trkPars);
  timeTracklet = evaluateTask(&Vertexer::findTrackletsHybrid, "Hybrid Vertexer tracklet finding", [](std::string) {});
  nTracklets01 = mTimeFrame->getTotalTrackletsTF(0);
  nTracklets12 = mTimeFrame->getTotalTrackletsTF(1);
  timeSelection = evaluateTask(&Vertexer::validateTrackletsHybrid, "Hybrid Vertexer adjacent tracklets validation", [](std::string) {});
  nSelectedTracklets = mTimeFrame->getNLinesTotal();
  timeVertexing = evaluateTask(&Vertexer::findVerticesHybrid, "Hybrid Vertexer vertex finding", [](std::string) {});
  printEpilog(logger, false, nTracklets01, nTracklets12, nSelectedTracklets, timeInit, timeTracklet, timeSelection, timeVertexing);
  return timeInit + timeTracklet + timeSelection + timeVertexing;
}

void Vertexer::getGlobalConfiguration()
{
  auto& vc = o2::its::VertexerParamConfig::Instance();
  vc.printKeyValues(true, true);
  auto& grc = o2::its::GpuRecoParamConfig::Instance();

  VertexingParameters verPar;
  verPar.deltaRof = vc.deltaRof;
  verPar.allowSingleContribClusters = vc.allowSingleContribClusters;
  verPar.zCut = vc.zCut;
  verPar.phiCut = vc.phiCut;
  verPar.pairCut = vc.pairCut;
  verPar.clusterCut = vc.clusterCut;
  verPar.histPairCut = vc.histPairCut;
  verPar.tanLambdaCut = vc.tanLambdaCut;
  verPar.lowMultBeamDistCut = vc.lowMultBeamDistCut;
  verPar.vertNsigmaCut = vc.vertNsigmaCut;
  verPar.vertRadiusSigma = vc.vertRadiusSigma;
  verPar.trackletSigma = vc.trackletSigma;
  verPar.maxZPositionAllowed = vc.maxZPositionAllowed;
  verPar.clusterContributorsCut = vc.clusterContributorsCut;
  verPar.maxTrackletsPerCluster = vc.maxTrackletsPerCluster;
  verPar.phiSpan = vc.phiSpan;
  verPar.nThreads = vc.nThreads;
  verPar.ZBins = vc.ZBins;
  verPar.PhiBins = vc.PhiBins;

  TimeFrameGPUParameters tfGPUpar;
  // tfGPUpar.nROFsPerChunk = grc.nROFsPerChunk;

  mTraits->updateVertexingParameters(verPar, tfGPUpar);
}

void Vertexer::adoptTimeFrame(TimeFrame& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

void Vertexer::printEpilog(std::function<void(std::string s)> logger,
                           bool isHybrid,
                           const unsigned int trackletN01, const unsigned int trackletN12, const unsigned selectedN,
                           const float initT, const float trackletT, const float selecT, const float vertexT)
{
  float total = initT + trackletT + selecT + vertexT;
  logger(fmt::format(" - {}Vertexer: trackleting found {}|{} tracklets, completed in: {} ms", isHybrid ? "Hybrid" : "", trackletN01, trackletN12, trackletT));
  logger(fmt::format(" - {}Vertexer: selected {} tracklets, completed in: {} ms", isHybrid ? "Hybrid" : "", selectedN, selecT));
  logger(fmt::format(" - {}Vertexer: vertexing completed in: {} ms", isHybrid ? "Hybrid" : "", vertexT));
  logger(fmt::format(" - Timeframe {} vertexing completed in: {} ms, using {} thread(s).", mTimeFrameCounter++, total, mTraits->getNThreads()));
}

} // namespace its
} // namespace o2
