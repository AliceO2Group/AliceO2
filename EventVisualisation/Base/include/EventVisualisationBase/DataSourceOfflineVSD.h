//
// Created by jmy on 26.02.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINEVSD_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINEVSD_H

#include <EventVisualisationBase/DataSourceOffline.h>
#include <TString.h>
#include <TEveTrack.h>
#include <TEveViewer.h>
#include <TEveVSD.h>




namespace o2  {
namespace event_visualisation {


class DataSourceOfflineVSD : public DataSourceOffline {
public:
    TEveVSD *fVSD;       // Visualisation Summary Data


    TFile *fFile;
    TObjArray *fEvDirKeys;
    Int_t fMaxEv, fCurEv;

    // ----------------------------------------------------------
    // Event visualization structures
    // ----------------------------------------------------------

    TEveTrackList *fTrackList;
    TEvePointSet *fITSClusters;
    TEvePointSet *fTPCClusters;
    TEvePointSet *fTRDClusters;
    TEvePointSet *fTOFClusters;
    TDirectory *fDirectory;

    DataSourceOfflineVSD();

    void LoadClusters(TEvePointSet *&ps, const TString &det_name, Int_t det_id);

    ~DataSourceOfflineVSD() override;

    void AttachEvent();

    TEveViewerList *viewers = nullptr;  // for debug purpose
    void DropEvent();

    void LoadEsdTracks();

    void open(TString ESDFileName) override;

    Bool_t GotoEvent(Int_t ev);
};




}
}
#endif