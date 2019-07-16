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

    TFile *fFile;
    TObjArray *fEvDirKeys;
    Int_t fMaxEv, fCurEv;

public:
    // ----------------------------------------------------------
    // Event visualization structures
    // ----------------------------------------------------------


    Int_t GetEventCount() override { return fEvDirKeys->GetEntriesFast(); };
    DataSourceOfflineVSD();
    ~DataSourceOfflineVSD() override;
    void open(TString ESDFileName) override;
    TObject* getEventData(int no) override;
};


}
}
#endif