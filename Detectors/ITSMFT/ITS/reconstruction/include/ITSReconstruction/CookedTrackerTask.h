// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CookedTrackerTask.h
/// \brief Definition of the ITS "Cooked Matrix" tracker task
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_COOKEDTRACKERTASK_H
#define ALICEO2_ITS_COOKEDTRACKERTASK_H

#include "FairTask.h" 

#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CookedTracker.h"

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
class CookedTrackerTask : public FairTask
{
 public:
  CookedTrackerTask(Int_t nThreads=1, Bool_t useMCTruth=kTRUE);
  ~CookedTrackerTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;
  void setBz(Double_t bz) { mTracker.setBz(bz); }

  void setContinuousMode(bool mode) { mContinuousMode = mode; }
  bool getContinuousMode() { return mContinuousMode; }
  
 private:
  bool mContinuousMode = true; ///< triggered or cont. mode
  CookedTracker mTracker; ///< Track finder

  const
  std::vector<o2::ITSMFT::Cluster>* mClustersArray=nullptr;   ///< Array of clusters
  const
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *mClsLabels=nullptr; ///< Cluster MC labels

  std::vector<CookedTrack> *mTracksArray=nullptr; ///< Array of tracks
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *mTrkLabels=nullptr; ///< Track MC labels

  ClassDefOverride(CookedTrackerTask, 1)
};
}
}

#endif /* ALICEO2_ITS_COOKEDTRACKERTASK */
