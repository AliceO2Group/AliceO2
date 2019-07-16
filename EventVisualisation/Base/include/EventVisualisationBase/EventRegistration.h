//
// Created by jmy on 15.07.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_EVENTREGISTRATION_H
#define ALICE_O2_EVENTVISUALISATION_BASE_EVENTREGISTRATION_H

#include <TEveElement.h>

namespace o2 {
namespace event_visualisation {

class EventRegistration {
private:
    static EventRegistration* instance;
public:
    /// Registers an element to be drawn
    virtual void registerElement(TEveElement *event) = 0;

    /// Removes all shapes representing current event
    virtual void destroyAllEvents() = 0;

    static EventRegistration* getInstance() { return instance;}
    static void setInstance(EventRegistration* instance) { EventRegistration::instance = instance;}
};

}
}
#endif //ALICE_O2_EVENTVISUALISATION_BASE_EVENTREGISTRATION_H
