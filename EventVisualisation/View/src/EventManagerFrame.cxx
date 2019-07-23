// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EventManagerFrame.cxx
/// \brief GUI (bottom buttons) for visualisation
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#include <TGButton.h>
#include <TGNumberEntry.h>
#include <TGLabel.h>
#include <EventVisualisationView/EventManagerFrame.h>
#include <EventVisualisationView/MultiView.h>
#include <EventVisualisationBase/DataSourceOffline.h>
#include <EventVisualisationBase/DataReaderVSD.h>
#include <Rtypes.h>
#include <iostream>


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
    fM->GotoEvent(0);
    fEventId->SetIntNumber(fM->getCurrentEvent());
}

void EventManagerFrame::DoPrevEvent() {
    fM->PrevEvent();
    fEventId->SetIntNumber(fM->getCurrentEvent());
}

void EventManagerFrame::DoNextEvent() {
    fM->NextEvent();
    fEventId->SetIntNumber(fM->getCurrentEvent());
}

void EventManagerFrame::DoLastEvent() {
    fM->GotoEvent(-1);  /// -1 means last available
    fEventId->SetIntNumber(fM->getCurrentEvent());
}

void EventManagerFrame::DoSetEvent() {
}

void EventManagerFrame::DoScreenshot() {
}


}
}
