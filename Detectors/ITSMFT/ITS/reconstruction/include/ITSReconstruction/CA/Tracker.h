// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Tracker.h
/// \brief 
///

#ifndef TRACKINGITSU_INCLUDE_TRACKER_H_
#define TRACKINGITSU_INCLUDE_TRACKER_H_

#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "DetectorsBase/Track.h"
#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/MathUtils.h"
#include "ITSReconstruction/CA/PrimaryVertexContext.h"
#include "ITSReconstruction/CA/Road.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace ITS
{
namespace CA
{

template<bool IsGPU>
class TrackerTraits
{
  public:
    void computeLayerTracklets(PrimaryVertexContext&);
    void computeLayerCells(PrimaryVertexContext&);

  protected:
    ~TrackerTraits() = default;
};

template<bool IsGPU>
class Tracker: private TrackerTraits<IsGPU>
{
  private:
    typedef TrackerTraits<IsGPU> Trait;
  public:
    Tracker(const Event &event);

    Tracker(const Tracker&) = delete;
    Tracker &operator=(const Tracker&) = delete;
    void setTrackMCTruthContainer(dataformats::MCTruthContainer<MCCompLabel> *trk);

    std::vector<std::vector<Road>> clustersToTracks();
    std::vector<std::vector<Road>> clustersToTracksVerbose();
    std::vector<std::vector<Road>> clustersToTracksMemoryBenchmark(std::ofstream&);
    std::vector<std::vector<Road>> clustersToTracksTimeBenchmark(std::ostream&);

  private:
    void computeTracklets();
    void computeCells();
    void findCellsNeighbours();
    void findRoads();
    void traverseCellsTree(const int, const int);
    void computeMontecarloLabels();
    void findTracks();
    Base::Track::TrackParCov buildTrackSeed(const Cluster& c1, const Cluster& c2, const Cluster& c3, const TrackingFrameInfo& tf3);

    void evaluateTask(void (Tracker<IsGPU>::*)(void), const char*);
    void evaluateTask(void (Tracker<IsGPU>::*)(void), const char*, std::ostream&);

    PrimaryVertexContext mPrimaryVertexContext;
    const Event &mEvent;
    dataformats::MCTruthContainer<o2::MCCompLabel> *mTrkLabels = nullptr; /// Track MC labels
};

template<> void TrackerTraits<TRACKINGITSU_GPU_MODE>::computeLayerTracklets(PrimaryVertexContext&);
template<> void TrackerTraits<TRACKINGITSU_GPU_MODE>::computeLayerCells(PrimaryVertexContext&);

template<bool IsGPU>
void Tracker<IsGPU>::setTrackMCTruthContainer(dataformats::MCTruthContainer<MCCompLabel> *trk) {
  mTrkLabels = trk;
}

}
}
}

#endif /* TRACKINGITSU_INCLUDE_TRACKER_H_ */
