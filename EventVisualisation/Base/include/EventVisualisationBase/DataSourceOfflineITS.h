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

    DataSourceOfflineITS();

    void open(TString fileName) override;

    Bool_t GotoEvent(Int_t ev);
};

}
}

#endif //O2EVE_DATASOURCEOFFLINEITS_H
