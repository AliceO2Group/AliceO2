// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CA/TrackerTask.h
/// \brief Definition of the ITS "Cellular Automaton" tracker task

#ifndef ALICEO2_ITS_CA_TRACKERTASK_H
#define ALICEO2_ITS_CA_TRACKERTASK_H

#include "FairTask.h"

#include <vector>

#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/Track.h"
#include "ITSReconstruction/CA/Tracker.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
  template<typename T>
  class MCTruthContainer;
}

namespace ITS
{
namespace CA
{

class TrackerTask : public FairTask
{
 public:
  TrackerTask(bool useMCTruth=true);
  ~TrackerTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;
  void setBz(double bz) { mTracker.setBz(bz); }

 private:
  Tracker<false> mTracker; ///< Track finder

  Event mEvent;
  const
  std::vector<ITSMFT::Cluster>* mClustersArray=nullptr;   ///< Array of clusters
  const
  dataformats::MCTruthContainer<MCCompLabel> *mClsLabels=nullptr; ///< Cluster MC labels

  std::vector<Track> *mTracksArray=nullptr; ///< Array of tracks
  dataformats::MCTruthContainer<MCCompLabel> *mTrkLabels=nullptr; ///< Track MC labels

  ClassDefOverride(TrackerTask, 1)
};

}
}
}

#endif /* ALICEO2_ITS_CA_TRACKERTASK */
