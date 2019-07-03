//
// Created by jmy on 23.02.19.
//


#include <TGButton.h>
#include <TGNumberEntry.h>
#include <TGLabel.h>
#include <EventVisualisationView/EventManagerFrame.h>
#include <EventVisualisationView/MultiView.h>
#include <EventVisualisationBase/DataSourceOffline.h>
#include <Rtypes.h>
#include <iostream>

//ClassImp(o2::event_visualisation::EventManagerFrame)
//using namespace ROOT;


ClassImp(o2::event_visualisation::EventManagerFrame)

namespace o2 {
namespace event_visualisation {


EventManagerFrame::EventManagerFrame(o2::event_visualisation::EventManager& eventManager)
:TGMainFrame(gClient->GetRoot(), 400, 100, kVerticalFrame) {
    fM = &eventManager;

    const TString cls("o2::event_visualisation::EventManagerFrame");
    TGTextButton *b = 0;
    TGHorizontalFrame *f = new TGHorizontalFrame(this);
    {
        Int_t width = 50;
        this->AddFrame(f, new TGLayoutHints(kLHintsExpandX, 0, 0, 2, 2));


        fFirstEvent = b = EventManagerFrame::makeButton(f, "First", width);
        b->Connect("Clicked()", cls, this, "DoFirstEvent()");
        fPrevEvent = b = EventManagerFrame::makeButton(f, "Prev", width);
        b->Connect("Clicked()", cls, this, "DoPrevEvent()");

        fEventId = new TGNumberEntry(f, 0, 5, -1, TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
                                     TGNumberFormat::kNELLimitMinMax, 0, 10000);
        f->AddFrame(fEventId, new TGLayoutHints(kLHintsNormal, 10, 5, 0, 0));
        fEventId->Connect("ValueSet(Long_t)", cls, this, "DoSetEvent()");
        fInfoLabel = new TGLabel(f);
        f->AddFrame(fInfoLabel, new TGLayoutHints(kLHintsNormal, 5, 10, 4, 0));

        fNextEvent = b = EventManagerFrame::makeButton(f, "Next", width);
        b->Connect("Clicked()", cls, this, "DoNextEvent()");
        fLastEvent = b = EventManagerFrame::makeButton(f, "Last", width);
        b->Connect("Clicked()", cls, this, "DoLastEvent()");
        fScreenshot = b = EventManagerFrame::makeButton(f, "Screenshot", 2 * width);
        b->Connect("Clicked()", cls, this, "DoScreenshot()");

    }

    SetCleanup(kDeepCleanup);

    Layout();
    MapSubwindows();
    MapWindow();
}

EventManagerFrame::~EventManagerFrame() {

}

TGTextButton* EventManagerFrame::makeButton(TGCompositeFrame *p, const char *txt,
        Int_t width, Int_t lo, Int_t ro, Int_t to, Int_t bo) {
    TGTextButton* b = new TGTextButton(p, txt);

    //b->SetFont("-adobe-helvetica-bold-r-*-*-48-*-*-*-*-*-iso8859-1");

    if (width > 0) {
        b->SetWidth(width);
        b->ChangeOptions(b->GetOptions() | kFixedWidth);
    }
    p->AddFrame(b, new TGLayoutHints(kLHintsNormal, lo,ro,to,bo));
    return b;
}

void EventManagerFrame::DoFirstEvent() {
    std::cout << "DoFirstEvent" << std::endl;
    fEventId->SetIntNumber(fM->gotoEvent(0));
}

void EventManagerFrame::DoPrevEvent() {
    std::cout << "DoPrevEvent" << std::endl;
    fEventId->SetIntNumber(fM->gotoEvent(fEventId->GetNumber()-1));
}

void EventManagerFrame::DoNextEvent() {
    std::cout << "DoNextEvent" << std::endl;
    fEventId->SetIntNumber(fM->gotoEvent(fEventId->GetNumber()+1));
}

void EventManagerFrame::DoLastEvent() {
    std::cout << "DoLastEvent" << std::endl;
    fEventId->SetIntNumber(fM->gotoEvent(-1));      // -1 means last available
}

void EventManagerFrame::DoSetEvent() {
}

void EventManagerFrame::DoScreenshot() {

}


}
}
