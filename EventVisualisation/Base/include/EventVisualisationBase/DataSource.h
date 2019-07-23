// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataReaderITS.h
/// \brief VSD specific reading from file(s) (Visualisation Summary Data)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch


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
