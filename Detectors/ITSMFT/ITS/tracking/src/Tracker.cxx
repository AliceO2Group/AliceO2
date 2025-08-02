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
#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackingConfigParam.h"

#include "ReconstructionDataFormats/Track.h"
#include <cassert>
#include <format>
#include <cstdlib>
#include <string>
#include <climits>

namespace o2::its
{
using o2::its::constants::GB;

Tracker::Tracker(TrackerTraits7* traits) : mTraits(traits)
{
  /// Initialise standard configuration with 1 iteration
  mTrkParams.resize(1);
  if (traits->isGPU()) {
    ITSGpuTrackingParamConfig::Instance().maybeOverride();
    ITSGpuTrackingParamConfig::Instance().printKeyValues(true, true);
  }
}

void Tracker::clustersToTracks(const LogFunc& logger, const LogFunc& error)
{
  LogFunc evalLog = [](const std::string&) {};

  double total{0};
  mTraits->updateTrackingParameters(mTrkParams);
  int maxNvertices{-1};
  if (mTrkParams[0].PerPrimaryVertexProcessing) {
    for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
      int minRof = o2::gpu::CAMath::Max(0, iROF - mTrkParams[0].DeltaROF);
      int maxRof = o2::gpu::CAMath::Min(mTimeFrame->getNrof(), iROF + mTrkParams[0].DeltaROF);
      maxNvertices = std::max(maxNvertices, (int)mTimeFrame->getPrimaryVertices(minRof, maxRof).size());
    }
  }

  int iteration{0}, iROFs{0}, iVertex{0};
  auto handleException = [&](const auto& err) {
    LOGP(error, "Too much memory used during {} in iteration {} in ROF span {}-{} iVtx={}: {:.2f} GB. Current limit is {:.2f} GB, check the detector status and/or the selections.",
         StateNames[mCurState], iteration, iROFs, iROFs + mTrkParams[iteration].nROFsPerIterations, iVertex,
         (double)mTimeFrame->getArtefactsMemory() / GB, (double)mTrkParams[iteration].MaxMemory / GB);
    LOGP(error, "Exception: {}", err.what());
    if (mTrkParams[iteration].DropTFUponFailure) {
      mMemoryPool->print();
      mTimeFrame->wipe();
      ++mNumberOfDroppedTFs;
      error("...Dropping Timeframe...");
    } else {
      throw err;
    }
  };

  try {
    for (iteration = 0; iteration < (int)mTrkParams.size(); ++iteration) {
      mMemoryPool->setMaxMemory(mTrkParams[iteration].MaxMemory);
      if (iteration == 3 && mTrkParams[0].DoUPCIteration) {
        mTimeFrame->swapMasks();
      }
      double timeTracklets{0.}, timeCells{0.}, timeNeighbours{0.}, timeRoads{0.};
      int nTracklets{0}, nCells{0}, nNeighbours{0}, nTracks{-static_cast<int>(mTimeFrame->getNumberOfTracks())};
      int nROFsIterations = mTrkParams[iteration].nROFsPerIterations > 0 ? mTimeFrame->getNrof() / mTrkParams[iteration].nROFsPerIterations + bool(mTimeFrame->getNrof() % mTrkParams[iteration].nROFsPerIterations) : 1;
      iVertex = std::min(maxNvertices, 0);
      logger(std::format("==== ITS {} Tracking iteration {} summary ====", mTraits->getName(), iteration));

      total += evaluateTask(&Tracker::initialiseTimeFrame, StateNames[mCurState = TFInit], iteration, logger, iteration);
      do {
        for (iROFs = 0; iROFs < nROFsIterations; ++iROFs) {
          timeTracklets += evaluateTask(&Tracker::computeTracklets, StateNames[mCurState = Trackleting], iteration, evalLog, iteration, iROFs, iVertex);
          nTracklets += mTraits->getTFNumberOfTracklets();
          float trackletsPerCluster = mTraits->getTFNumberOfClusters() > 0 ? float(mTraits->getTFNumberOfTracklets()) / float(mTraits->getTFNumberOfClusters()) : 0.f;
          if (trackletsPerCluster > mTrkParams[iteration].TrackletsPerClusterLimit) {
            error(std::format("Too many tracklets per cluster ({}) in iteration {} in ROF span {}-{}:, check the detector status and/or the selections. Current limit is {}",
                              trackletsPerCluster, iteration, iROFs, iROFs + mTrkParams[iteration].nROFsPerIterations, mTrkParams[iteration].TrackletsPerClusterLimit));
            break;
          }
          timeCells += evaluateTask(&Tracker::computeCells, StateNames[mCurState = Celling], iteration, evalLog, iteration);
          nCells += mTraits->getTFNumberOfCells();
          float cellsPerCluster = mTraits->getTFNumberOfClusters() > 0 ? float(mTraits->getTFNumberOfCells()) / float(mTraits->getTFNumberOfClusters()) : 0.f;
          if (cellsPerCluster > mTrkParams[iteration].CellsPerClusterLimit) {
            error(std::format("Too many cells per cluster ({}) in iteration {} in ROF span {}-{}, check the detector status and/or the selections. Current limit is {}",
                              cellsPerCluster, iteration, iROFs, iROFs + mTrkParams[iteration].nROFsPerIterations, mTrkParams[iteration].CellsPerClusterLimit));
            break;
          }
          timeNeighbours += evaluateTask(&Tracker::findCellsNeighbours, StateNames[mCurState = Neighbouring], iteration, evalLog, iteration);
          nNeighbours += mTimeFrame->getNumberOfNeighbours();
          timeRoads += evaluateTask(&Tracker::findRoads, StateNames[mCurState = Roading], iteration, evalLog, iteration);
        }
      } while (++iVertex < maxNvertices);
      logger(std::format(" - Tracklet finding: {} tracklets found in {:.2f} ms", nTracklets, timeTracklets));
      logger(std::format(" - Cell finding: {} cells found in {:.2f} ms", nCells, timeCells));
      logger(std::format(" - Neighbours finding: {} neighbours found in {:.2f} ms", nNeighbours, timeNeighbours));
      logger(std::format(" - Track finding: {} tracks found in {:.2f} ms", nTracks + mTimeFrame->getNumberOfTracks(), timeRoads));
      total += timeTracklets + timeCells + timeNeighbours + timeRoads;
      if (mTraits->supportsExtendTracks() && mTrkParams[iteration].UseTrackFollower) {
        int nExtendedTracks{-mTimeFrame->mNExtendedTracks}, nExtendedClusters{-mTimeFrame->mNExtendedUsedClusters};
        auto timeExtending = evaluateTask(&Tracker::extendTracks, "Extending tracks", iteration, evalLog, iteration);
        total += timeExtending;
        logger(std::format(" - Extending Tracks: {} extended tracks using {} clusters found in {:.2f} ms", nExtendedTracks + mTimeFrame->mNExtendedTracks, nExtendedClusters + mTimeFrame->mNExtendedUsedClusters, timeExtending));
      }
      if (mTrkParams[iteration].PrintMemory) {
        mMemoryPool->print();
      }
    }
    if (mTraits->supportsFindShortPrimaries() && mTrkParams[0].FindShortTracks) {
      auto nTracksB = mTimeFrame->getNumberOfTracks();
      total += evaluateTask(&Tracker::findShortPrimaries, "Short primaries finding", 0, logger);
      auto nTracksA = mTimeFrame->getNumberOfTracks();
      logger(std::format("  `-> found {} additional tracks", nTracksA - nTracksB));
    }
    if constexpr (constants::DoTimeBenchmarks) {
      logger(std::format("=== TimeFrame {} processing completed in: {:.2f} ms using {} thread(s) ===", mTimeFrameCounter, total, mTraits->getNThreads()));
    }
  } catch (const BoundedMemoryResource::MemoryLimitExceeded& err) {
    handleException(err);
    return;
  } catch (const std::bad_alloc& err) {
    handleException(err);
    return;
  } catch (...) {
    error("Uncaught exception, all bets are off...");
  }

  if (mTimeFrame->hasMCinformation()) {
    computeTracksMClabels();
  }
  rectifyClusterIndices();
  ++mTimeFrameCounter;
  mTotalTime += total;

  if (mTrkParams[0].PrintMemory) {
    mTimeFrame->printArtefactsMemory();
    mMemoryPool->print();
  }
}

void Tracker::computeRoadsMClabels()
{
  /// Moore's Voting Algorithm
  if (!mTimeFrame->hasMCinformation()) {
    return;
  }

  mTimeFrame->initialiseRoadLabels();

  int roadsNum{static_cast<int>(mTimeFrame->getRoads().size())};

  for (int iRoad{0}; iRoad < roadsNum; ++iRoad) {

    Road<5>& currentRoad{mTimeFrame->getRoads()[iRoad]};
    std::vector<std::pair<MCCompLabel, size_t>> occurrences;
    bool isFakeRoad{false};
    bool isFirstRoadCell{true};

    for (int iCell{0}; iCell < mTrkParams[0].CellsPerRoad(); ++iCell) {
      const int currentCellIndex{currentRoad[iCell]};

      if (currentCellIndex == constants::UnusedIndex) {
        if (isFirstRoadCell) {
          continue;
        } else {
          break;
        }
      }

      const CellSeed& currentCell{mTimeFrame->getCells()[iCell][currentCellIndex]};

      if (isFirstRoadCell) {

        const int cl0index{mTimeFrame->getClusters()[iCell][currentCell.getFirstClusterIndex()].clusterId};
        auto cl0labs{mTimeFrame->getClusterLabels(iCell, cl0index)};
        bool found{false};
        for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
          std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
          for (const auto& label : cl0labs) {
            if (label == occurrence.first) {
              ++occurrence.second;
              found = true;
              // break; // uncomment to stop to the first hit
            }
          }
        }
        if (!found) {
          for (const auto& label : cl0labs) {
            occurrences.emplace_back(label, 1);
          }
        }

        const int cl1index{mTimeFrame->getClusters()[iCell + 1][currentCell.getSecondClusterIndex()].clusterId};

        const auto& cl1labs{mTimeFrame->getClusterLabels(iCell + 1, cl1index)};
        found = false;
        for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
          std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
          for (auto& label : cl1labs) {
            if (label == occurrence.first) {
              ++occurrence.second;
              found = true;
              // break; // uncomment to stop to the first hit
            }
          }
        }
        if (!found) {
          for (auto& label : cl1labs) {
            occurrences.emplace_back(label, 1);
          }
        }

        isFirstRoadCell = false;
      }

      const int cl2index{mTimeFrame->getClusters()[iCell + 2][currentCell.getThirdClusterIndex()].clusterId};
      const auto& cl2labs{mTimeFrame->getClusterLabels(iCell + 2, cl2index)};
      bool found{false};
      for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
        std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
        for (auto& label : cl2labs) {
          if (label == occurrence.first) {
            ++occurrence.second;
            found = true;
            // break; // uncomment to stop to the first hit
          }
        }
      }
      if (!found) {
        for (auto& label : cl2labs) {
          occurrences.emplace_back(label, 1);
        }
      }
    }

    std::sort(occurrences.begin(), occurrences.end(), [](auto e1, auto e2) {
      return e1.second > e2.second;
    });

    auto maxOccurrencesValue = occurrences[0].first;
    mTimeFrame->setRoadLabel(iRoad, maxOccurrencesValue.getRawValue(), isFakeRoad);
  }
}

void Tracker::computeTracksMClabels()
{
  for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
    for (auto& track : mTimeFrame->getTracks(iROF)) {
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
      mTimeFrame->getTracksLabel(iROF).emplace_back(maxOccurrencesValue);
    }
  }
}

void Tracker::rectifyClusterIndices()
{
  for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
    for (auto& track : mTimeFrame->getTracks(iROF)) {
      for (int iCluster = 0; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
        const int index = track.getClusterIndex(iCluster);
        if (index != constants::UnusedIndex) {
          track.setExternalClusterIndex(iCluster, mTimeFrame->getClusterExternalIndex(iCluster, index));
        }
      }
    }
  }
}

void Tracker::adoptTimeFrame(TimeFrame7& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

void Tracker::printSummary() const
{
  auto avgTF = mTotalTime * 1.e-3 / ((mTimeFrameCounter > 0) ? (double)mTimeFrameCounter : -1.0);
  auto avgTFwithDropped = mTotalTime * 1.e-3 / (((mTimeFrameCounter + mNumberOfDroppedTFs) > 0) ? (double)(mTimeFrameCounter + mNumberOfDroppedTFs) : -1.0);
  LOGP(info, "Tracker summary: Processed {} TFs (dropped {}) in TOT={:.2f} s, AVG/TF={:.2f} ({:.2f}) s", mTimeFrameCounter, mNumberOfDroppedTFs, mTotalTime * 1.e-3, avgTF, avgTFwithDropped);
}

} // namespace o2::its
