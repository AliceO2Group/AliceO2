// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackerTask.h
/// \brief Task driving the track finding from MFT clusters
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 7, 2018

#ifndef ALICEO2_MFT_TRACKERTASK_H_
#define ALICEO2_MFT_TRACKERTASK_H_

#include "FairTask.h"

#include "DataFormatsMFT/TrackMFT.h"
#include "MFTBase/GeometryTGeo.h"
#include "MFTReconstruction/Tracker.h"

namespace o2
{
class MCCompLabel;

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace mft
{
class TrackerTask : public FairTask
{
 public:
  TrackerTask(Int_t nThreads = 1, Bool_t useMCTruth = kTRUE);
  ~TrackerTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;

  void setContinuousMode(bool mode) { mContinuousMode = mode; }
  bool getContinuousMode() { return mContinuousMode; }

 private:
  bool mContinuousMode = true; ///< triggered or cont. mode
  Tracker mTracker;            ///< Track finder

  const std::vector<o2::itsmft::Cluster>* mClustersArray = nullptr;               ///< Array of clusters
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; ///< Cluster MC labels

  std::vector<TrackMFT>* mTracksArray = nullptr;                            ///< Array of tracks
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTrkLabels = nullptr; ///< Track MC labels

  ClassDefOverride(TrackerTask, 1);
};
} // namespace mft
} // namespace o2

#endif
