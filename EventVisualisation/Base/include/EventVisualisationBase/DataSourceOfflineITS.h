//
// Created by jmy on 09.07.19.
//

#ifndef O2EVE_DATASOURCEOFFLINEITS_H
#define O2EVE_DATASOURCEOFFLINEITS_H

#include <EventVisualisationBase/DataSourceOffline.h>

namespace o2 {
namespace event_visualisation {



class DataSourceOfflineITS : public DataSourceOffline {
public:

    std::string name() override { return "DataSourceOfflineITS"; }
    TString digifile;
    TString clusfile;
    TString tracfile;
    int entry = 0;
    int chip = 13;
    DataSourceOfflineITS();

    bool open() override;

    TEveElementList* gotoEvent(Int_t event) override ;
};

}
}

#endif //O2EVE_DATASOURCEOFFLINEITS_H
