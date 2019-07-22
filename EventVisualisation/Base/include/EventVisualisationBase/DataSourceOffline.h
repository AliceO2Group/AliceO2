//
// Created by jmy on 26.02.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H

#include <EventVisualisationBase/DataSource.h>
#include <EventVisualisationBase/DataReader.h>
#include <TString.h>

namespace o2  {
namespace event_visualisation {

class DataSourceOffline : public DataSource {
protected:
    static DataReader* instance[EVisualisationGroup::NvisualisationGroups];
public:

    void nextEvent() override {};
    DataSourceOffline() {}

    ~DataSourceOffline() override {};

    DataSourceOffline(DataSourceOffline const&) = delete;
    /// Deleted assignemt operator
    void operator=(DataSourceOffline const&) = delete;

    Int_t GetEventCount() override {
      for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; i++)
        if(instance[i] != nullptr)
          return instance[i]->GetEventCount();
      return 1;
    };

    void registerReader(DataReader *reader, EVisualisationGroup purpose) {
      instance[purpose] = reader;
    };
    TObject* getEventData(int no, EVisualisationGroup purpose) override;
};


}
}


#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCEOFFLINE_H
