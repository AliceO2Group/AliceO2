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

#include "ITStracking/Cell.h"
#include "ITStracking/Constants.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Smoother.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackingConfigParam.h"

#include "ReconstructionDataFormats/Track.h"
#include <cassert>
#include <iostream>
#include <dlfcn.h>
#include <cstdlib>
#include <string>
#include <climits>

namespace o2
{
namespace its
{

Tracker::Tracker(o2::its::TrackerTraits* traits)
{
  /// Initialise standard configuration with 1 iteration
  mTrkParams.resize(1);
  mTraits = traits;
}

Tracker::~Tracker() = default;

void Tracker::clustersToTracks(std::function<void(std::string s)> logger, std::function<void(std::string s)> error)
{
  double total{0};
  mTraits->UpdateTrackingParameters(mTrkParams);
  int maxNvertices{-1};
  if (mTrkParams[0].PerPrimaryVertexProcessing) {
    for (int iROF{0}; iROF < mTimeFrame->getNrof(); ++iROF) {
      maxNvertices = std::max(maxNvertices, (int)mTimeFrame->getPrimaryVertices(iROF).size());
    }
  }

  for (int iteration = 0; iteration < (int)mTrkParams.size(); ++iteration) {

    logger(fmt::format("ITS Tracking iteration {} summary:", iteration));
    double timeTracklets{0.}, timeCells{0.}, timeNeighbours{0.}, timeRoads{0.};
    int nTracklets{0}, nCells{0}, nNeighbours{0}, nTracks{-static_cast<int>(mTimeFrame->getNumberOfTracks())};

    total += evaluateTask(&Tracker::initialiseTimeFrame, "Timeframe initialisation", logger, iteration);
    int nROFsIterations = mTrkParams[iteration].nROFsPerIterations > 0 ? mTimeFrame->getNrof() / mTrkParams[iteration].nROFsPerIterations + bool(mTimeFrame->getNrof() % mTrkParams[iteration].nROFsPerIterations) : 1;
    int iVertex{std::min(maxNvertices, 0)};

    do {
      for (int iROFs{0}; iROFs < nROFsIterations; ++iROFs) {
        timeTracklets += evaluateTask(
          &Tracker::computeTracklets, "Tracklet finding", [](std::string) {}, iteration, iROFs, iVertex);
        nTracklets += mTraits->getTFNumberOfTracklets();
        if (!mTimeFrame->checkMemory(mTrkParams[iteration].MaxMemory)) {
          error(fmt::format("Too much memory used during trackleting in iteration {}, check the detector status and/or the selections.", iteration));
          break;
        }
        float trackletsPerCluster = mTraits->getTFNumberOfClusters() > 0 ? float(mTraits->getTFNumberOfTracklets()) / mTraits->getTFNumberOfClusters() : 0.f;
        if (trackletsPerCluster > mTrkParams[iteration].TrackletsPerClusterLimit) {
          error(fmt::format("Too many tracklets per cluster ({}) in iteration {}, check the detector status and/or the selections. Current limit is {}", trackletsPerCluster, iteration, mTrkParams[iteration].TrackletsPerClusterLimit));
          break;
        }

        timeCells += evaluateTask(
          &Tracker::computeCells, "Cell finding", [](std::string) {}, iteration);
        nCells += mTraits->getTFNumberOfCells();
        if (!mTimeFrame->checkMemory(mTrkParams[iteration].MaxMemory)) {
          error(fmt::format("Too much memory used during cell finding in iteration {}, check the detector status and/or the selections.", iteration));
          break;
        }
        float cellsPerCluster = mTraits->getTFNumberOfClusters() > 0 ? float(mTraits->getTFNumberOfCells()) / mTraits->getTFNumberOfClusters() : 0.f;
        if (cellsPerCluster > mTrkParams[iteration].CellsPerClusterLimit) {
          error(fmt::format("Too many cells per cluster ({}) in iteration {}, check the detector status and/or the selections. Current limit is {}", cellsPerCluster, iteration, mTrkParams[iteration].CellsPerClusterLimit));
          break;
        }

        timeNeighbours += evaluateTask(
          &Tracker::findCellsNeighbours, "Neighbour finding", [](std::string) {}, iteration);
        nNeighbours += mTimeFrame->getNumberOfNeighbours();
        timeRoads += evaluateTask(
          &Tracker::findRoads, "Road finding", [](std::string) {}, iteration);
      }
      iVertex++;
    } while (iVertex < maxNvertices);
    logger(fmt::format("\t- Tracklet finding: {} tracklets in {:.2f} ms", nTracklets, timeTracklets));
    logger(fmt::format("\t- Cell finding: {} cells found in {:.2f} ms", nCells, timeCells));
    logger(fmt::format("\t- Neighbours finding: {} neighbours found in {:.2f} ms", nNeighbours, timeNeighbours));
    logger(fmt::format("\t- Track finding: {} tracks found in {:.2f} ms", nTracks + mTimeFrame->getNumberOfTracks(), timeRoads));
    total += timeTracklets + timeCells + timeNeighbours + timeRoads;
    total += evaluateTask(&Tracker::extendTracks, "Extending tracks", logger, iteration);
  }

  total += evaluateTask(&Tracker::findShortPrimaries, "Short primaries finding", logger);

  std::stringstream sstream;
  if (constants::DoTimeBenchmarks) {
    sstream << std::setw(2) << " - "
            << "Timeframe " << mTimeFrameCounter++ << " processing completed in: " << total << "ms using " << mTraits->getNThreads() << " threads.";
  }
  logger(sstream.str());

  if (mTimeFrame->hasMCinformation()) {
    computeTracksMClabels();
  }
  rectifyClusterIndices();
  mNumberOfRuns++;
}

void Tracker::clustersToTracksHybrid(std::function<void(std::string s)> logger, std::function<void(std::string s)> error)
{
  double total{0.};
  mTraits->UpdateTrackingParameters(mTrkParams);
  for (int iteration = 0; iteration < (int)mTrkParams.size(); ++iteration) {
    LOGP(info, "Iteration {}", iteration);
    total += evaluateTask(&Tracker::initialiseTimeFrameHybrid, "Hybrid Timeframe initialisation", logger, iteration);
    total += evaluateTask(&Tracker::computeTrackletsHybrid, "Hybrid Tracklet finding", logger, iteration, iteration, iteration); // TODO: iteration argument put just for the sake of the interface, to be updated with the proper ROF slicing
    logger(fmt::format("\t- Number of tracklets: {}", mTraits->getTFNumberOfTracklets()));
    if (!mTimeFrame->checkMemory(mTrkParams[iteration].MaxMemory)) {
      error("Too much memory used during trackleting, check the detector status and/or the selections.");
      break;
    }
    float trackletsPerCluster = mTraits->getTFNumberOfClusters() > 0 ? float(mTraits->getTFNumberOfTracklets()) / mTraits->getTFNumberOfClusters() : 0.f;
    if (trackletsPerCluster > mTrkParams[iteration].TrackletsPerClusterLimit) {
      error(fmt::format("Too many tracklets per cluster ({}), check the detector status and/or the selections. Current limit is {}", trackletsPerCluster, mTrkParams[iteration].TrackletsPerClusterLimit));
      break;
    }

    total += evaluateTask(&Tracker::computeCellsHybrid, "Hybrid Cell finding", logger, iteration);
    logger(fmt::format("\t- Number of Cells: {}", mTraits->getTFNumberOfCells()));
    if (!mTimeFrame->checkMemory(mTrkParams[iteration].MaxMemory)) {
      error("Too much memory used during cell finding, check the detector status and/or the selections.");
      break;
    }
    float cellsPerCluster = mTraits->getTFNumberOfClusters() > 0 ? float(mTraits->getTFNumberOfCells()) / mTraits->getTFNumberOfClusters() : 0.f;
    if (cellsPerCluster > mTrkParams[iteration].CellsPerClusterLimit) {
      error(fmt::format("Too many cells per cluster ({}), check the detector status and/or the selections. Current limit is {}", cellsPerCluster, mTrkParams[iteration].CellsPerClusterLimit));
      break;
    }
    total += evaluateTask(&Tracker::findCellsNeighboursHybrid, "Hybrid Neighbour finding", logger, iteration);
    logger(fmt::format("\t- Number of Neighbours: {}", mTimeFrame->getNumberOfNeighbours()));
    total += evaluateTask(&Tracker::findRoadsHybrid, "Hybrid Track finding", logger, iteration);
    logger(fmt::format("\t- Number of Tracks: {}", mTimeFrame->getNumberOfTracks()));
    // total += evaluateTask(&Tracker::findTracksHybrid, "Hybrid Track fitting", logger, iteration);
  }
}

void Tracker::initialiseTimeFrame(int& iteration)
{
  mTraits->initialiseTimeFrame(iteration);
}

void Tracker::computeTracklets(int& iteration, int& iROFslice, int& iVertex)
{
  mTraits->computeLayerTracklets(iteration, iROFslice, iVertex);
}

void Tracker::computeCells(int& iteration)
{
  mTraits->computeLayerCells(iteration);
}

void Tracker::findCellsNeighbours(int& iteration)
{
  mTraits->findCellsNeighbours(iteration);
}

void Tracker::findRoads(int& iteration)
{
  mTraits->findRoads(iteration);
}

void Tracker::initialiseTimeFrameHybrid(int& iteration)
{
  mTraits->initialiseTimeFrameHybrid(iteration);
}

void Tracker::computeTrackletsHybrid(int& iteration, int&, int&)
{
  mTraits->computeTrackletsHybrid(iteration, iteration, iteration); // placeholder for the proper ROF/vertex slicing
}

void Tracker::computeCellsHybrid(int& iteration)
{
  mTraits->computeCellsHybrid(iteration);
}

void Tracker::findCellsNeighboursHybrid(int& iteration)
{
  mTraits->findCellsNeighboursHybrid(iteration);
}

void Tracker::findRoadsHybrid(int& iteration)
{
  mTraits->findRoadsHybrid(iteration);
}

void Tracker::findTracksHybrid(int& iteration)
{
  mTraits->findTracksHybrid(iteration);
}

void Tracker::findTracks()
{
  mTraits->findTracks();
}

void Tracker::extendTracks(int& iteration)
{
  mTraits->extendTracks(iteration);
}

void Tracker::findShortPrimaries()
{
  mTraits->findShortPrimaries();
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

      if (currentCellIndex == constants::its::UnusedIndex) {
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
          for (auto& label : cl0labs) {
            if (label == occurrence.first) {
              ++occurrence.second;
              found = true;
              // break; // uncomment to stop to the first hit
            }
          }
        }
        if (!found) {
          for (auto& label : cl0labs) {
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
        if (index == constants::its::UnusedIndex) {
          continue;
        }
        auto labels = mTimeFrame->getClusterLabels(iCluster, index);
        bool found{false};
        for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
          std::pair<o2::MCCompLabel, size_t>& occurrence = occurrences[iOcc];
          for (auto& label : labels) {
            if (label == occurrence.first) {
              ++occurrence.second;
              found = true;
              // break; // uncomment to stop to the first hit
            }
          }
        }
        if (!found) {
          for (auto& label : labels) {
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
        if (clid != constants::its::UnusedIndex) {
          auto labelsSpan = mTimeFrame->getClusterLabels(ic, clid);
          for (auto& currentLabel : labelsSpan) {
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
        if (index != constants::its::UnusedIndex) {
          track.setExternalClusterIndex(iCluster, mTimeFrame->getClusterExternalIndex(iCluster, index));
        }
      }
    }
  }
}

void Tracker::getGlobalConfiguration()
{
  auto& tc = o2::its::TrackerParamConfig::Instance();
  if (tc.useMatCorrTGeo) {
    mTraits->setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrTGeo);
  } else if (tc.useFastMaterial) {
    mTraits->setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE);
  } else {
    mTraits->setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
  }
  setNThreads(tc.nThreads);
  int nROFsPerIterations = tc.nROFsPerIterations > 0 ? tc.nROFsPerIterations : -1;
  if (tc.nOrbitsPerIterations > 0) {
    /// code to be used when the number of ROFs per orbit is known, this gets priority over the number of ROFs per iteration
  }
  for (auto& params : mTrkParams) {
    if (params.NLayers == 7) {
      for (int i{0}; i < 7; ++i) {
        params.SystErrorY2[i] = tc.sysErrY2[i] > 0 ? tc.sysErrY2[i] : params.SystErrorY2[i];
        params.SystErrorZ2[i] = tc.sysErrZ2[i] > 0 ? tc.sysErrZ2[i] : params.SystErrorZ2[i];
      }
    }
    params.DeltaROF = tc.deltaRof;
    params.MaxChi2ClusterAttachment = tc.maxChi2ClusterAttachment > 0 ? tc.maxChi2ClusterAttachment : params.MaxChi2ClusterAttachment;
    params.MaxChi2NDF = tc.maxChi2NDF > 0 ? tc.maxChi2NDF : params.MaxChi2NDF;
    params.PhiBins = tc.LUTbinsPhi > 0 ? tc.LUTbinsPhi : params.PhiBins;
    params.ZBins = tc.LUTbinsZ > 0 ? tc.LUTbinsZ : params.ZBins;
    params.PVres = tc.pvRes > 0 ? tc.pvRes : params.PVres;
    params.NSigmaCut *= tc.nSigmaCut > 0 ? tc.nSigmaCut : 1.f;
    params.CellDeltaTanLambdaSigma *= tc.deltaTanLres > 0 ? tc.deltaTanLres : 1.f;
    params.TrackletMinPt *= tc.minPt > 0 ? tc.minPt : 1.f;
    params.nROFsPerIterations = nROFsPerIterations;
    params.PerPrimaryVertexProcessing = tc.perPrimaryVertexProcessing;
    for (int iD{0}; iD < 3; ++iD) {
      params.Diamond[iD] = tc.diamondPos[iD];
    }
    params.UseDiamond = tc.useDiamond;
    if (tc.maxMemory) {
      params.MaxMemory = tc.maxMemory;
    }
    if (tc.useTrackFollower >= 0) {
      params.UseTrackFollower = tc.useTrackFollower;
    }
    if (tc.cellsPerClusterLimit >= 0) {
      params.CellsPerClusterLimit = tc.cellsPerClusterLimit;
    }
    if (tc.trackletsPerClusterLimit >= 0) {
      params.TrackletsPerClusterLimit = tc.trackletsPerClusterLimit;
    }
    if (tc.findShortTracks >= 0) {
      params.FindShortTracks = tc.findShortTracks;
    }
  }
}

void Tracker::adoptTimeFrame(TimeFrame& tf)
{
  mTimeFrame = &tf;
  mTraits->adoptTimeFrame(&tf);
}

void Tracker::setBz(float bz)
{
  mTraits->setBz(bz);
}

void Tracker::setCorrType(const o2::base::PropagatorImpl<float>::MatCorrType type)
{
  mTraits->setCorrType(type);
}

bool Tracker::isMatLUT() const
{
  return mTraits->isMatLUT();
}

void Tracker::setNThreads(int n)
{
  mTraits->setNThreads(n);
}

int Tracker::getNThreads() const
{
  return mTraits->getNThreads();
}
} // namespace its
} // namespace o2
