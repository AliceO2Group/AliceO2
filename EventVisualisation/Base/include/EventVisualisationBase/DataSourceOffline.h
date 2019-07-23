// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSourceOffline.h
/// \brief Grouping reading from file(s)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

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
