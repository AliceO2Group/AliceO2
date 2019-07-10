//
// Created by jmy on 25.06.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERITS_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERITS_H

/// DataInterpreterVSD prepares random events
///
/// This class overrides DataInterpreter and implements method
/// returning visualisation objects representing data from ITS file
/// with tracks colored by PID only.

#include "EventVisualisationBase/DataInterpreter.h"
#include "EventVisualisationBase/EventManager.h"
#include "EventVisualisationBase/VisualisationConstants.h"

namespace o2 {
namespace event_visualisation {


class DataInterpreterITS : public DataInterpreter {
public:
    // Default constructor
    DataInterpreterITS();

    // Default destructor
    ~DataInterpreterITS() final;

    // Returns a list of random tracks colored by PID
    TEveElement *interpretDataForType(EDataType type) final;
};

}
}

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERITS_H
