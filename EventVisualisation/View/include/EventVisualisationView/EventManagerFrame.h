//
// Created by jmy on 23.02.19.
//

#ifndef ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
#define ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H

#include <TGMdiMainFrame.h>
#include "EventVisualisationBase/EventManager.h"

class TGTextButton;
class TGCompositeFrame;
class TGNumberEntry;
class TGLabel;

namespace o2 {
namespace event_visualisation {



class EventManagerFrame : public TGMainFrame {
private:
    static TGTextButton* makeButton(TGCompositeFrame* p, const char* txt, Int_t width=0,
                              Int_t lo=0, Int_t ro=0, Int_t to=0, Int_t bo=0);
protected:
    //TEveEventManager   *fM;            // Model object.
    EventManager   *fM;            // Model object.

    TGTextButton   *fFirstEvent;   // Go to first event
    TGTextButton   *fPrevEvent;    // Go to prev event
    TGTextButton   *fNextEvent;    // Go to next event
    TGTextButton   *fLastEvent;    // Go to last event
    TGTextButton         *fScreenshot;   // Save screenshot to file
    TGNumberEntry        *fEventId;      // Display/edit current event id
    TGLabel              *fInfoLabel;    // Display last available event id
public:
    EventManagerFrame(o2::event_visualisation::EventManager& eventManager);
    virtual ~EventManagerFrame();
    ClassDef(EventManagerFrame, 0); // GUI window for AliEveEventManager.


public: // slots
    void DoFirstEvent();
    void DoPrevEvent();
    void DoNextEvent();
    void DoLastEvent();
    void DoSetEvent();
    void DoScreenshot();
};


}
}

#endif //ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
