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
/// \brief Definition of the ITS CA tracker task

#ifndef O2_ITSMFT_RECONSTRUCTION_CA_TRACKERTASK_H_
#define O2_ITSMFT_RECONSTRUCTION_CA_TRACKERTASK_H_

#include "FairTask.h"

#include <memory>
#include <vector>

#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/Tracker.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace ITS
{
namespace CA
{

class TrackerTask : public FairTask
{
 public:
  TrackerTask(bool useMCTruth=true);
  ~TrackerTask() override {}

  InitStatus Init() override;
  void Exec(Option_t* option) override;

 private:
  GeometryTGeo*  mGeometry; ///< ITS geometry
  Event          mEvent;    ///< CA tracker event
  Tracker<false> mTracker;  ///< Track finder

  const std::vector<o2::ITSMFT::Cluster>* mClustersArray=nullptr;             ///< Array of clusters
  const dataformats::MCTruthContainer<o2::MCCompLabel> *mClsLabels=nullptr;   ///< Cluster MC labels

  std::vector<Track>* mTracksArray=nullptr;                           ///< Array of tracks
  dataformats::MCTruthContainer<o2::MCCompLabel>* mTrkLabels=nullptr; ///< Track MC labels

  ClassDefOverride(TrackerTask, 1)
};
}
}
}

#endif /* O2_ITSMFT_RECONSTRUCTION_CA_TRACKERTASK_H_ */
