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
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/Constants.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackingConfigParam.h"

#include <cassert>
#include <format>
#include <cstdlib>
#include <string>

namespace o2::its
{
using o2::its::constants::GB;

template <int NLayers>
Tracker<NLayers>::Tracker(TrackerTraits<NLayers>* traits) : mTraits(traits)
{
  /// Initialise standard configuration with 1 iteration
  mTrkParams.resize(1);
  if (traits->isGPU()) {
    ITSGpuTrackingParamConfig::Instance().maybeOverride();
    ITSGpuTrackingParamConfig::Instance().printKeyValues(true, true);
  }
}

template <int NLayers>
void Tracker<NLayers>::clustersToTracks(const LogFunc& logger, const LogFunc& error)
{
  LogFunc evalLog = [](const std::string&) {};

  double total{0};
  mTraits->updateTrackingParameters(mTrkParams);
  mTimeFrame->updateROFVertexLookupTable();

  int maxNvertices{-1};
  if (mTrkParams[0].PerPrimaryVertexProcessing) {
    maxNvertices = mTimeFrame->getROFVertexLookupTableView().getMaxVerticesPerROF();
  }

  int iteration{0}, iVertex{0};
  auto handleException = [&](const auto& err) {
    LOGP(error, "Too much memory in {} in iteration {} iVtx={}: {:.2f} GB. Current limit is {:.2f} GB, check the detector status and/or the selections.",
         StateNames[mCurState], iteration, iVertex,
         (double)mTimeFrame->getArtefactsMemory() / GB,
         (double)mTrkParams[iteration].MaxMemory / GB);
    if (typeid(err) != typeid(std::bad_alloc)) { // only print if the exceptions is different from what is expected
      LOGP(error, "Exception: {}", err.what());
    }
    if (mTrkParams[iteration].DropTFUponFailure) {
      mMemoryPool->print();
      mTimeFrame->wipe();
      ++mNumberOfDroppedTFs;
      error(std::format("...Dropping TimeSlice {} (out of {} dropped {})...", mTimeSlice, mTimeFrameCounter, mNumberOfDroppedTFs));
    } else {
      throw err;
    }
  };

  try {
    for (iteration = 0; iteration < (int)mTrkParams.size(); ++iteration) {
      mMemoryPool->setMaxMemory(mTrkParams[iteration].MaxMemory);
      if (iteration == 3 && mTrkParams[0].DoUPCIteration) {
        mTimeFrame->useUPCMask();
      }
      float timeTracklets{0.}, timeCells{0.}, timeNeighbours{0.}, timeRoads{0.};
      size_t nTracklets{0}, nCells{0}, nNeighbours{0};
      int nTracks{-static_cast<int>(mTimeFrame->getNumberOfTracks())};
      iVertex = std::min(maxNvertices, 0);
      logger(std::format("==== ITS {} Tracking iteration {} summary ====", mTraits->getName(), iteration));
      total += evaluateTask(&Tracker::initialiseTimeFrame, StateNames[mCurState = TFInit], iteration, logger, iteration);
      do {
        timeTracklets += evaluateTask(&Tracker::computeTracklets, StateNames[mCurState = Trackleting], iteration, evalLog, iteration, iVertex);
        nTracklets += mTraits->getTFNumberOfTracklets();
        timeCells += evaluateTask(&Tracker::computeCells, StateNames[mCurState = Celling], iteration, evalLog, iteration);
        nCells += mTraits->getTFNumberOfCells();
        timeNeighbours += evaluateTask(&Tracker::findCellsNeighbours, StateNames[mCurState = Neighbouring], iteration, evalLog, iteration);
        nNeighbours += mTimeFrame->getNumberOfNeighbours();
        timeRoads += evaluateTask(&Tracker::findRoads, StateNames[mCurState = Roading], iteration, evalLog, iteration);
      } while (++iVertex < maxNvertices);
      logger(std::format(" - Tracklet finding: {} tracklets found in {:.2f} ms", nTracklets, timeTracklets));
      logger(std::format(" - Cell finding: {} cells found in {:.2f} ms", nCells, timeCells));
      logger(std::format(" - Neighbours finding: {} neighbours found in {:.2f} ms", nNeighbours, timeNeighbours));
      logger(std::format(" - Track finding: {} tracks found in {:.2f} ms", nTracks + mTimeFrame->getNumberOfTracks(), timeRoads));
      total += timeTracklets + timeCells + timeNeighbours + timeRoads;
      if (mTrkParams[iteration].PrintMemory) {
        mMemoryPool->print();
      }
    }
    if constexpr (constants::DoTimeBenchmarks) {
      logger(std::format("=== TimeSlice {} processing completed in: {:.2f} ms using {} thread(s) ===", mTimeSlice, total, mTraits->getNThreads()));
    }
  } catch (const BoundedMemoryResource::MemoryLimitExceeded& err) {
    handleException(err);
    return;
  } catch (const std::bad_alloc& err) {
    handleException(err);
    return;
  } catch (const std::exception& err) {
    error(std::format("Uncaught exception, all bets are off... {}", err.what()));
    // clear tracks explicitly since if not fatalising on exception this may contain partial output
    mTimeFrame->getTracks().clear();
    return;
  }

  if (mTimeFrame->hasMCinformation()) {
    computeTracksMClabels();
  }
  rectifyClusterIndices();
  sortTracks();

  ++mTimeFrameCounter;
  mTotalTime += total;

  if (mTrkParams[0].PrintMemory) {
    mTimeFrame->printArtefactsMemory();
    mMemoryPool->print();
  }
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
void Tracker<NLayers>::printSummary() const
{
  auto avgTF = mTotalTime * 1.e-3 / ((mTimeFrameCounter > 0) ? (double)mTimeFrameCounter : -1.0);
  auto avgTFwithDropped = mTotalTime * 1.e-3 / (((mTimeFrameCounter + mNumberOfDroppedTFs) > 0) ? (double)(mTimeFrameCounter + mNumberOfDroppedTFs) : -1.0);
  LOGP(info, "Tracker summary: Processed {} TFs (dropped {}) in TOT={:.2f} s, AVG/TF={:.2f} ({:.2f}) s", mTimeFrameCounter, mNumberOfDroppedTFs, mTotalTime * 1.e-3, avgTF, avgTFwithDropped);
}

template class Tracker<7>;
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class Tracker<11>;
#endif

} // namespace o2::its
