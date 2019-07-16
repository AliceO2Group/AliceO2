//
// Created by jmy on 25.06.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERVSD_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERVSD_H

/// DataInterpreterVSD prepares random events
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

namespace o2 {
namespace event_visualisation {


class DataInterpreterVSD : public DataInterpreter {
private:
  void LoadClusters(TEvePointSet *&ps, const TString &det_name, Int_t det_id);
  void AttachEvent();

  TEveViewerList *viewers = nullptr;  // for debug purpose


  void LoadEsdTracks();
  TEveTrackList *fTrackList = nullptr;
  TEvePointSet *fITSClusters = nullptr;
  TEvePointSet *fTPCClusters = nullptr;
  TEvePointSet *fTRDClusters = nullptr;
  TEvePointSet *fTOFClusters = nullptr;
  TDirectory *fDirectory = nullptr;
  TEveVSD *fVSD = nullptr;       // Visualisation Summary Data
public:
  void DropEvent();
    // Default constructor
    DataInterpreterVSD();

    // Default destructor
    virtual ~DataInterpreterVSD() final;

    // Returns a list of random tracks colored by PID
    TEveElement *interpretDataForType(TObject* data, EDataType type) final;
};

}
}

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERVSD_H
