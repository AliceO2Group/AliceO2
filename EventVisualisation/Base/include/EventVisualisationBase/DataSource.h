//
// Created by jmy on 26.02.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H

#include <EventVisualisationBase/VisualisationConstants.h>
#include <TQObject.h>


namespace o2 {
namespace event_visualisation {

class DataSource : public TQObject {
public:

    virtual void nextEvent() {};
    virtual TObject* getEventData(int no, EVisualisationGroup purpose) { return nullptr;};
    virtual Int_t GetEventCount() { return 0; };
    virtual void open() {};
    virtual void close() {};

    DataSource(){};

    /// Default destructor
    virtual ~DataSource(){};

    /// Deleted copy constructor
    DataSource(DataSource const &) = delete;

    /// Deleted assignemt operator
    void operator=(DataSource const &) = delete;
};

}
}

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
