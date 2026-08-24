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
/// \file Tracker.cxx
/// \brief
///

#include "ITStracking/Tracker.h"
#include "ITSMFTTracking/BoundedAllocator.h"
#include "ITSMFTTracking/Constants.h"
#include "ITStracking/TrackerTraits.h"
#include "ITSMFTTracking/ITSTrackingConfigParam.h"

#include <cassert>
#include <algorithm>
#include <limits>
#include <format>
#include <cstdlib>
#include <string>

namespace o2::its
{
using o2::its::constants::GB;

template <int NLayers>
Tracker<NLayers>::Tracker(TrackerTraits<NLayers>* traits) : mTraits(traits)
{
}

template <int NLayers>
float Tracker<NLayers>::clustersToVertices(const LogFunc& logger)
{
  mTraits->updateTrackingParameters(mTrkParams);
  bounded_vector<int> seedingIts(mMemoryPool.get());
  for (int i = 0; i < (int)mTrkParams.size(); ++i) {
    if (mTrkParams[i].PassFlags[IterationStep::SeedingVertexPass]) {
      seedingIts.push_back(i);
    }
  }
  if (seedingIts.empty()) {
    return 0.f;
  }
  logger(std::format("==== ITS {} Seeding-vertex pass ====", mTraits->getName()));
  float total{0.f};
  for (size_t sp = 0; sp < seedingIts.size(); ++sp) {
    const int it = seedingIts[sp];
    mMemoryPool->setMaxMemory(mTrkParams[it].MaxMemory);
    const bool bootstrapBeam = !mTimeFrame->isBeamOverridden();
    const int maxBootstrap = (sp == 0 && bootstrapBeam) ? constants::MaxBootstrapPasses : 1;
    if (sp > 0) {
      ROFMaskTable<NLayers> upcMask{mTimeFrame->getROFOverlapTable()};
      const auto& vtxLookup = mTimeFrame->getROFVertexLookupTableView();
      const int threshold = mTrkParams[it].VertPerRofThreshold;
      int nEnabled = 0, nTotal = 0;
      for (int layer = 0; layer < NLayers; ++layer) {
        const int nRofs = mTimeFrame->getROFOverlapTableView().getLayer(layer).mNROFsTF;
        for (int rof = 0; rof < nRofs; ++rof) {
          // Same predicate as skipROF(), applied earlier.
          const bool empty = (int)vtxLookup.getVertices(layer, rof).getEntries() <= threshold;
          upcMask.setROFEnabled(layer, rof, empty ? 1u : 0u);
          if (layer == 1) {
            nTotal++;
            nEnabled += empty;
          }
        }
      }
      mTimeFrame->setSeedingUPCMask(std::move(upcMask));
      mTimeFrame->useSeedingUPCMask();
      logger(std::format(" - Seeding pass {} (iteration {}): UPC pass on {}/{} ROFs without vertices",
                         sp, it, nEnabled, nTotal));
    }
    for (int pass = 0; pass < maxBootstrap; ++pass) {
      const float prevBeamX = mTimeFrame->getBeamX();
      const float prevBeamY = mTimeFrame->getBeamY();
      total += evaluateTask(&Tracker::initialiseTimeFrame, StateNames[mCurStep = TFInit], it, logger, it);
      total += evaluateTask(&Tracker::computeTracklets, StateNames[mCurStep = Trackleting], it, logger, it, -1);
      const int nTracklets = mTraits->getTFNumberOfTracklets();
      total += evaluateTask(&Tracker::computeCells, StateNames[mCurStep = Celling], it, logger, it);
      total += evaluateTask(&Tracker::computeVertexCandidates, StateNames[mCurStep = CellLinearising], it, logger, it);
      logger(std::format(" - Seeding pass {}.{}: {} tracklets, {} cells, {} lines", sp, pass,
                         nTracklets, mTraits->getTFNumberOfCells(),
                         mTimeFrame->getNLinesTotal()));
      total += evaluateTask(&Tracker::computeVertices, StateNames[mCurStep = SeedingVertices], it, logger, it);
      if (sp > 0 || !bootstrapBeam) {
        continue;
      }
      total += evaluateTask(&Tracker::computeBeamFromVertices, StateNames[mCurStep = BeamPositioning], it, logger, it);
      const float dx = mTimeFrame->getBeamX() - prevBeamX;
      const float dy = mTimeFrame->getBeamY() - prevBeamY;
      if (dx * dx + dy * dy < constants::BeamConvergence2) {
        logger(std::format(" - Beam bootstrap converged after pass {} (beam shift < 50 um)", pass));
        break;
      }
    }
  }
  if (seedingIts.size() > 1) {
    mTimeFrame->useMultiplictyMask();
  }
  return total;
}

template <int NLayers>
float Tracker<NLayers>::clustersToTracks(const LogFunc& logger, const LogFunc& error)
{
  LogFunc evalLog = [](const std::string&) {};

  float total{0};
  mTraits->updateTrackingParameters(mTrkParams);

  int maxNvertices{-1};
  int firstTrackingIteration{0};
  while (firstTrackingIteration < (int)mTrkParams.size() &&
         mTrkParams[firstTrackingIteration].PassFlags[IterationStep::SeedingVertexPass]) {
    ++firstTrackingIteration;
  }
  if (firstTrackingIteration < (int)mTrkParams.size() && mTrkParams[firstTrackingIteration].PerPrimaryVertexProcessing) {
    maxNvertices = mTimeFrame->getROFVertexLookupTableView().getMaxVerticesPerROF();
  }

  int iteration{0}, iVertex{0};
  auto handleException = [&](const auto& err) {
    if (mTrkParams[iteration].MaxMemory == std::numeric_limits<size_t>::max()) {
      LOGP(error, "Allocation failed in {} in iteration {} iVtx={} ({:.2f} GB of host artefacts, no host limit set), check the detector status and/or the selections.",
           StateNames[mCurStep], iteration, iVertex,
           (double)mTimeFrame->getArtefactsMemory() / GB);
    } else {
      LOGP(error, "Too much memory in {} in iteration {} iVtx={}: {:.2f} GB. Current limit is {:.2f} GB, check the detector status and/or the selections.",
           StateNames[mCurStep], iteration, iVertex,
           (double)mTimeFrame->getArtefactsMemory() / GB,
           (double)mTrkParams[iteration].MaxMemory / GB);
    }
    if (typeid(err) != typeid(std::bad_alloc)) { // only print if the exceptions is different from what is expected
      LOGP(error, "Exception: {}", err.what());
    }
    if (mTrkParams[iteration].DropTFUponFailure) {
      mMemoryPool->print();
      mTimeFrame->wipe();
      mTimeFrame->getCapacityEstimator().reset();
      ++mNumberOfDroppedTFs;
      error(std::format("...Dropping TimeSlice {} (out of {} dropped {})...", mTimeSlice, mTimeFrameCounter, mNumberOfDroppedTFs));
    } else {
      throw err;
    }
  };

  try {
    for (iteration = 0; iteration < (int)mTrkParams.size(); ++iteration) {
      mMemoryPool->setMaxMemory(mTrkParams[iteration].MaxMemory);
      if (mTrkParams[iteration].PassFlags[IterationStep::UseUPCMask]) {
        mTimeFrame->useUPCMask();
      }
      if (mTrkParams[iteration].PassFlags[IterationStep::SeedingVertexPass]) {
        continue; // the seeding-vertex pass runs as its own phase (clustersToVertices), not as a tracking iteration
      }
      float timeFrame{0.}, timeTracklets{0.}, timeCells{0.}, timeNeighbours{0.}, timeRoads{0.};
      size_t nTracklets{0}, nCells{0}, nNeighbours{0};
      int nTracks{-static_cast<int>(mTimeFrame->getNumberOfTracks())};
      iVertex = std::min(maxNvertices, 0);
      logger(std::format("==== ITS {} Tracking iteration {} summary ====", mTraits->getName(), iteration));
      total += timeFrame = evaluateTask(&Tracker::initialiseTimeFrame, StateNames[mCurStep = TFInit], iteration, evalLog, iteration);
      logger(std::format(" - TimeFrame initialisation completed in {:.2f} ms", timeFrame));
      do {
        timeTracklets += evaluateTask(&Tracker::computeTracklets, StateNames[mCurStep = Trackleting], iteration, evalLog, iteration, iVertex);
        nTracklets += mTraits->getTFNumberOfTracklets();
        timeCells += evaluateTask(&Tracker::computeCells, StateNames[mCurStep = Celling], iteration, evalLog, iteration);
        nCells += mTraits->getTFNumberOfCells();
        timeNeighbours += evaluateTask(&Tracker::findCellsNeighbours, StateNames[mCurStep = Neighbouring], iteration, evalLog, iteration);
        nNeighbours += mTimeFrame->getNumberOfNeighbours();
        timeRoads += evaluateTask(&Tracker::findRoads, StateNames[mCurStep = Roading], iteration, evalLog, iteration);
      } while (++iVertex < maxNvertices);
      logger(std::format(" - Tracklet finding: {} tracklets found in {:.2f} ms", nTracklets, timeTracklets));
      logger(std::format(" - Cell finding: {} cells found in {:.2f} ms", nCells, timeCells));
      logger(std::format(" - Neighbours finding: {} neighbours found in {:.2f} ms", nNeighbours, timeNeighbours));
      logger(std::format(" - Track finding: {} tracks found in {:.2f} ms", nTracks + mTimeFrame->getNumberOfTracks(), timeRoads));
      if (mTrkParams[iteration].PassFlags[IterationStep::TrackFollowerTop] || mTrkParams[iteration].PassFlags[IterationStep::TrackFollowerBot]) {
        logger(std::format(" - Integrated track extension: {} tracks accepted using {} clusters", mTimeFrame->getNExtendedTracks(), mTimeFrame->getNExtendedClusters()));
      }
      total += timeTracklets + timeCells + timeNeighbours + timeRoads;
    }
  } catch (const BoundedMemoryResource::MemoryLimitExceeded& err) {
    handleException(err);
    return -1.f;
  } catch (const std::bad_alloc& err) {
    handleException(err);
    return -1.f;
  } catch (const std::exception& err) {
    error(std::format("Uncaught exception, all bets are off... {}", err.what()));
    // clear tracks explicitly since if not fatalising on exception this may contain partial output
    mTimeFrame->getTracks().clear();
    return -1.f;
  }

  if (mTimeFrame->hasMCinformation()) {
    computeTracksMClabels();
  }
  rectifyClusterIndices();
  sortTracks();

  ++mTimeFrameCounter;
  mTotalTime += total;

  return total;
}

template <int NLayers>
void Tracker<NLayers>::computeTracksMClabels()
{
  for (auto& track : mTimeFrame->getTracks()) {
    std::vector<std::pair<MCCompLabel, size_t>> occurrences;
    occurrences.clear();

    for (int iCluster = 0; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
      const int index = track.getClusterIndex(iCluster);
      if (index == constants::UnusedIndex) {
        continue;
      }
      auto labels = mTimeFrame->getClusterLabels(iCluster, index);
      bool found{false};
      for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
        std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
        for (const auto& label : labels) {
          if (label == occurrence.first) {
            ++occurrence.second;
            found = true;
            // break; // uncomment to stop to the first hit
          }
        }
      }
      if (!found) {
        for (const auto& label : labels) {
          occurrences.emplace_back(label, 1);
        }
      }
    }
    std::sort(std::begin(occurrences), std::end(occurrences), [](auto e1, auto e2) {
      return e1.second > e2.second;
    });

    auto maxOccurrencesValue = occurrences[0].first;
    uint32_t pattern = track.getPattern();
    // set fake clusters pattern
    for (int ic{TrackITSExt::MaxClusters}; ic--;) {
      auto clid = track.getClusterIndex(ic);
      if (clid != constants::UnusedIndex) {
        auto labelsSpan = mTimeFrame->getClusterLabels(ic, clid);
        for (const auto& currentLabel : labelsSpan) {
          if (currentLabel == maxOccurrencesValue) {
            pattern |= 0x1 << (16 + ic); // set bit if correct
            break;
          }
        }
      }
    }
    track.setPattern(pattern);
    if (occurrences[0].second < track.getNumberOfClusters()) {
      maxOccurrencesValue.setFakeFlag();
    }
    mTimeFrame->getTracksLabel().emplace_back(maxOccurrencesValue);
  }
}

template <int NLayers>
void Tracker<NLayers>::rectifyClusterIndices()
{
  for (auto& track : mTimeFrame->getTracks()) {
    for (int iCluster = 0; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
      const int index = track.getClusterIndex(iCluster);
      if (index != constants::UnusedIndex) {
        track.setExternalClusterIndex(iCluster, mTimeFrame->getClusterExternalIndex(iCluster, index));
      }
    }
  }
}

template <int NLayers>
void Tracker<NLayers>::sortTracks()
{
  auto& trks = mTimeFrame->getTracks();
  bounded_vector<size_t> indices(trks.size(), mMemoryPool.get());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&trks](size_t i, size_t j) {
    // provide tracks sorted by lower-bound
    const auto& a = trks[i];
    const auto& b = trks[j];
    const auto aLower = a.getTimeStamp().getTimeStamp() - a.getTimeStamp().getTimeStampError();
    const auto bLower = b.getTimeStamp().getTimeStamp() - b.getTimeStamp().getTimeStampError();
    if (aLower != bLower) {
      return aLower < bLower;
    }
    return a.isBetter(b, 1e9); // then sort tracks in quality
  });
  bounded_vector<TrackITSExt> sortedTrks(mMemoryPool.get());
  sortedTrks.reserve(trks.size());
  for (size_t idx : indices) {
    sortedTrks.push_back(trks[idx]);
  }
  trks.swap(sortedTrks);
  if (mTimeFrame->hasMCinformation()) {
    auto& trksLabels = mTimeFrame->getTracksLabel();
    bounded_vector<MCCompLabel> sortedLabels(mMemoryPool.get());
    sortedLabels.reserve(trksLabels.size());
    for (size_t idx : indices) {
      sortedLabels.push_back(trksLabels[idx]);
    }
    trksLabels.swap(sortedLabels);
  }
}

template <int NLayers>
void Tracker<NLayers>::adoptTimeFrame(TimeFrame<NLayers>& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

template <int NLayers>
void Tracker<NLayers>::addTimingStatCurStep(int iteration, double timeMs)
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
void Tracker<NLayers>::printSummary() const
{
  auto avgTF = mTotalTime * 1.e-3 / ((mTimeFrameCounter > 0) ? (double)mTimeFrameCounter : -1.0);
  auto avgTFwithDropped = mTotalTime * 1.e-3 / (((mTimeFrameCounter + mNumberOfDroppedTFs) > 0) ? (double)(mTimeFrameCounter + mNumberOfDroppedTFs) : -1.0);
  LOGP(info, "Tracker summary: Processed {} TFs (dropped {}) in TOT={:.2f} s, AVG/TF={:.2f} ({:.2f}) s", mTimeFrameCounter, mNumberOfDroppedTFs, mTotalTime * 1.e-3, avgTF, avgTFwithDropped);
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

template class Tracker<7>;
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class Tracker<11>;
template class Tracker<13>;
#endif

} // namespace o2::its
