//
// Created by jmy on 26.02.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H

#include <TQObject.h>


namespace o2 {
namespace event_visualisation {

class DataSource : public TQObject {
public:
    virtual Int_t gotoEvent(Int_t event) {};
    virtual void nextEvent() {};
    virtual Int_t GetEventCount() { return 0; };
    virtual void open(TString ESDFileName) {};
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
