//
//  Created by jmy on 26.02.19.
//

#include <EventVisualisationBase/DataSourceOffline.h>



o2::event_visualisation::DataSourceOffline::DataSourceOffline(TString ESDFileName) {
    this->fgESDFileName = ESDFileName;
    open();
}

o2::event_visualisation::DataSourceOffline::~DataSourceOffline() {

}

int o2::event_visualisation::DataSourceOffline::gotoEvent(Int_t ev) {

    if (ev < 0 || ev >= esd.fMaxEv)
    {
        Warning("GotoEvent", "Invalid event id %d.", ev);
        return kFALSE;
    }

/*    DropEvent();

    // Connect to new event-data.

    fCurEv = ev;
    fDirectory = (TDirectory*) ((TKey*) fEvDirKeys->At(fCurEv))->ReadObj();
    fVSD->SetDirectory(fDirectory);

    AttachEvent();

    // Load event data into visualization structures.

    LoadClusters(fITSClusters, "ITS", 0);
    LoadClusters(fTPCClusters, "TPC", 1);
    LoadClusters(fTRDClusters, "TRD", 2);
    LoadClusters(fTOFClusters, "TOF", 3);

    LoadEsdTracks();

    // Fill projected views.

    TEveElement* top = gEve->GetCurrentEvent();

    gMultiView->DestroyEventRPhi();
    gMultiView->ImportEventRPhi(top);

    gMultiView->DestroyEventRhoZ();
    gMultiView->ImportEventRhoZ(top);

    gEve->Redraw3D(kFALSE, kTRUE);

    return kTRUE;

*/

    return ev;
}

void o2::event_visualisation::DataSourceOffline::open() {

}
