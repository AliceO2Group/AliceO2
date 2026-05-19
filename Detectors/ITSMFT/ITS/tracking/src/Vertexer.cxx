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
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/TrackingConfigParam.h"

namespace o2::its
{

template <int NLayers>
Vertexer<NLayers>::Vertexer(VertexerTraitsN* traits) : mTraits(traits)
{
  if (!mTraits) {
    LOG(fatal) << "nullptr passed to ITS vertexer construction.";
  }
  mVertParams.resize(1);
}

template <int NLayers>
float Vertexer<NLayers>::clustersToVertices(LogFunc logger)
{
  LogFunc evalLog = [](const std::string&) {};

  if (mTimeFrame->hasMCinformation() && mVertParams[0].useTruthSeeding) {
    float t = evaluateTask(&Vertexer::addTruthSeeds, StateNames[mCurStep = TruthSeeding], 0, evalLog);
    sortVertices();
    ++mTimeFrameCounter;
    return t;
  }

  TrackingParameters trkPars;
  mTraits->updateVertexingParameters(mVertParams);

  auto handleException = [&](const auto& err) {
    LOGP(error, "Encountered critical error in step {}, stopping further processing of this TF: {}", StateNames[mCurStep], err.what());
    if (!mVertParams[0].DropTFUponFailure) {
      throw err;
    } else {
      LOGP(error, "Dropping this TF!");
    }
  };

  float timeTracklet{0.f}, timeSelection{0.f}, timeVertexing{0.f}, timeInit{0.f};
  bool completed = false;
  try {
    for (int iteration = 0; iteration < (int)mVertParams.size(); ++iteration) {
      mMemoryPool->setMaxMemory(mVertParams[iteration].MaxMemory);
      unsigned int nTracklets01{0}, nTracklets12{0};
      logger(fmt::format("=== ITS {} Seeding vertexer iteration {} summary:", mTraits->getName(), iteration));
      const auto& currentVtxPars = mTraits->getVertexingParameters()[iteration];
      trkPars.PhiBins = currentVtxPars.PhiBins;
      trkPars.ZBins = currentVtxPars.ZBins;
      trkPars.LayerZ = currentVtxPars.LayerZ;
      trkPars.LayerRadii = currentVtxPars.LayerRadii;
      trkPars.PassFlags = mVertParams[iteration].PassFlags;
      trkPars.PassFlags.set(IterationStep::FirstPass, IterationStep::RebuildClusterLUT);
      auto timeInitIteration = evaluateTask(&Vertexer::initialiseVertexer, StateNames[mCurStep = Init], iteration, evalLog, trkPars);
      auto timeTrackletIteration = evaluateTask(&Vertexer::findTracklets, StateNames[mCurStep = Trackleting], iteration, evalLog, iteration);
      nTracklets01 = mTimeFrame->getTotalTrackletsTF(0);
      nTracklets12 = mTimeFrame->getTotalTrackletsTF(1);
      auto timeSelectionIteration = evaluateTask(&Vertexer::validateTracklets, StateNames[mCurStep = Selection], iteration, evalLog, iteration);
      const auto nVerticesBefore = mTimeFrame->getPrimaryVertices().size();
      auto timeVertexingIteration = evaluateTask(&Vertexer::findVertices, StateNames[mCurStep = Finding], iteration, evalLog, iteration);
      const auto nVerticesAfter = mTimeFrame->getPrimaryVertices().size();
      printEpilog(logger, nTracklets01, nTracklets12, mTimeFrame->getNLinesTotal(), nVerticesAfter - nVerticesBefore, nVerticesAfter, timeInitIteration, timeTrackletIteration, timeSelectionIteration, timeVertexingIteration);
      timeInit += timeInitIteration;
      timeTracklet += timeTrackletIteration;
      timeSelection += timeSelectionIteration;
      timeVertexing += timeVertexingIteration;

      // update LUT with all currently found vertices so in second iteration we can check vertPerROFThreshold
      sortVertices();
    }
    completed = true;
  } catch (const BoundedMemoryResource::MemoryLimitExceeded& err) {
    handleException(err);
  } catch (const std::bad_alloc& err) {
    handleException(err);
  } catch (...) {
    LOGP(fatal, "Uncaught exception!");
  }

  if (completed) {
    ++mTimeFrameCounter;
  }

  return timeInit + timeTracklet + timeSelection + timeVertexing;
}

template <int NLayers>
void Vertexer<NLayers>::sortVertices()
{
  auto& pvs = mTimeFrame->getPrimaryVertices();
  bounded_vector<size_t> indices(pvs.size(), mMemoryPool.get());
  std::iota(indices.begin(), indices.end(), 0);
  // provide vertices sorted by lower-bound
  std::sort(indices.begin(), indices.end(), [&pvs](size_t i, size_t j) {
    const auto& a = pvs[i].getTimeStamp();
    const auto& b = pvs[j].getTimeStamp();
    const auto aLower = a.lower();
    const auto bLower = b.lower();
    if (aLower != bLower) {
      return aLower < bLower;
    }
    return pvs[i].getNContributors() > pvs[j].getNContributors();
  });
  bounded_vector<Vertex> sortedVtx(mMemoryPool.get());
  sortedVtx.reserve(pvs.size());
  for (const size_t idx : indices) {
    sortedVtx.push_back(pvs[idx]);
  }
  pvs.swap(sortedVtx);
  if (mTimeFrame->hasMCinformation()) {
    auto& mc = mTimeFrame->getPrimaryVerticesLabels();
    bounded_vector<VertexLabel> sortedMC(mMemoryPool.get());
    for (const size_t idx : indices) {
      sortedMC.push_back(mc[idx]);
    }
    mc.swap(sortedMC);
  }
  // update LUT after sorting
  mTimeFrame->updateROFVertexLookupTable();
}

template <int NLayers>
void Vertexer<NLayers>::adoptTimeFrame(TimeFrameN& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

template <int NLayers>
void Vertexer<NLayers>::addTimingStatCurStep(int iteration, double timeMs)
{
  if (iteration < 0) {
    return;
  }
  if (mTimingStats.size() < (iteration + 1)) {
    mTimingStats.resize(iteration + 1);
  }
  mTimingStats[iteration][mCurStep].add(timeMs);
}

template <int NLayers>
void Vertexer<NLayers>::printSummary() const
{
  LOGP(info, "Vertexer summary: Processed {} TFs", mTimeFrameCounter);
  for (size_t iteration = 0; iteration < mTimingStats.size(); ++iteration) {
    for (size_t state = 0; state < NSteps; ++state) {
      const auto& stats = mTimingStats[iteration][state];
      if (!stats.calls) {
        continue;
      }
      LOGP(info, " - iter {} {}: calls={} total={:.2f} ms avg={:.2f} ms", iteration, StateNames[state], stats.calls, stats.totalTimeMs, stats.averageTimeMs());
    }
  }
}

template <int NLayers>
void Vertexer<NLayers>::printEpilog(LogFunc& logger,
                                    unsigned int trackletN01, unsigned int trackletN12,
                                    unsigned selectedN, unsigned int vertexN, unsigned int totalVertexN,
                                    float initT, float trackletT, float selecT, float vertexT)
{
  logger(fmt::format(" - {}: completed in {:.2f} ms", StateNames[Init], initT));
  logger(fmt::format(" - {}: found {} | {} tracklets in {:.2f} ms", StateNames[Trackleting], trackletN01, trackletN12, trackletT));
  logger(fmt::format(" - {}: selected {} tracklets in {:.2f} ms", StateNames[Selection], selectedN, selecT));
  logger(fmt::format(" - {}: found {} vertices (total {}) in {:.2f} ms", StateNames[Finding], vertexN, totalVertexN, vertexT));
}

template class Vertexer<7>;

} // namespace o2::its
