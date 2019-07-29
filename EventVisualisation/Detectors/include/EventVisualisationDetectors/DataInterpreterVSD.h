// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataInterpreterVSD.h
/// \brief converting VSD data to Event Visualisation primitives
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERVSD_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERVSD_H

///
/// This class overrides DataInterpreter and implements method
/// returning visualisation objects representing data from VSD file
/// with tracks colored by PID only.

#include "EventVisualisationBase/DataInterpreter.h"
#include "EventVisualisationBase/EventManager.h"
#include "EventVisualisationBase/VisualisationConstants.h"
#include <TEvePointSet.h>
#include <TEveViewer.h>
#include <TEveTrack.h>
#include <TEveVSD.h>

namespace o2
{
namespace event_visualisation
{

class DataInterpreterVSD : public DataInterpreter
{
 private:
  void LoadClusters(TEvePointSet*& ps, const TString& det_name, Int_t det_id);
  void AttachEvent();

  TEveViewerList* mViewers = nullptr; // for debug purpose

  void LoadEsdTracks();
  TEveTrackList* mTrackList = nullptr;
  TEvePointSet* mITSClusters = nullptr;
  TEvePointSet* mTPCClusters = nullptr;
  TEvePointSet* mTRDClusters = nullptr;
  TEvePointSet* mTOFClusters = nullptr;
  TDirectory* mDirectory = nullptr;
  TEveVSD* mVSD = nullptr; // Visualisation Summary Data
 public:
  void DropEvent();
  // Default constructor
  DataInterpreterVSD() = default;

  // Default destructor
  ~DataInterpreterVSD() final;

  // Returns a list of random tracks colored by PID
  TEveElement* interpretDataForType(TObject* data, EVisualisationDataType type) final;
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERVSD_H
