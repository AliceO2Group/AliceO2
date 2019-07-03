//
// Created by jmy on 26.02.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H

#include <EventVisualisationBase/DataSource.h>
#include <TString.h>

// VSD
#include <TEveVSD.h>

namespace o2  {
namespace event_visualisation {

class DataSourceOffline : public DataSource {
protected:
    TString fgESDFileName ;
    bool isOpen = kFALSE;
    void open();
    struct {
        int fMaxEv;
        TEveVSD    *fVSD;
    } esd;
public:
    int gotoEvent(Int_t event) override ;
    void nextEvent() override {};
    DataSourceOffline(TString ESDFileName);
    virtual ~DataSourceOffline() final;
    DataSourceOffline(DataSourceOffline const&) = delete;
    /// Deleted assignemt operator
    void operator=(DataSourceOffline const&) = delete;
};

}
}


#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H
