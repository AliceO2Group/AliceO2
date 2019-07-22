//
// Created by jmy on 26.02.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINEVSD_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINEVSD_H

#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationBase/VisualisationConstants.h>
#include <TString.h>
#include <TEveTrack.h>
#include <TEveViewer.h>
#include <TEveVSD.h>




namespace o2  {
namespace event_visualisation {


class DataReaderVSD : public DataReader {
    TFile *fFile;
    TObjArray *fEvDirKeys;
    Int_t fMaxEv, fCurEv;

public:
    Int_t GetEventCount() override { return fEvDirKeys->GetEntriesFast(); };
    DataReaderVSD();
    ~DataReaderVSD() override;
    void open() override;
    TObject* getEventData(int no) override;
};


}
}
#endif