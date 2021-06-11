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
#include <TTimer.h>
#include <EventVisualisationView/EventManagerFrame.h>
#include <EventVisualisationView/MultiView.h>
#include <EventVisualisationBase/DataSourceOffline.h>
#include <EventVisualisationDetectors/DataReaderVSD.h>
#include <Rtypes.h>
#include <mutex>

std::mutex mtx; // mutex for critical section

ClassImp(o2::event_visualisation::EventManagerFrame);

namespace o2
{
  namespace event_visualisation
  {

  EventManagerFrame::~EventManagerFrame()
  {
    this->StopTimer();
  }

  EventManagerFrame::EventManagerFrame(o2::event_visualisation::EventManager& eventManager)
    : TGMainFrame(gClient->GetRoot(), 400, 100, kVerticalFrame)
  {
    mEventManager = &eventManager;
    this->mTimer = new TTimer(); // Auto-load time in seconds
    this->mTime = 2;
    this->mTimer->Connect("Timeout()", "o2::event_visualisation::EventManagerFrame", this, "DoTimeTick()");

    const TString cls("o2::event_visualisation::EventManagerFrame");
    TGTextButton* b = nullptr;
    TGHorizontalFrame* f = new TGHorizontalFrame(this);
    {
      Int_t width = 50;
      this->AddFrame(f, new TGLayoutHints(kLHintsExpandX, 0, 0, 2, 2));

      b = EventManagerFrame::makeButton(f, "First", width);
      b->Connect("Clicked()", cls, this, "DoFirstEvent()");
      b = EventManagerFrame::makeButton(f, "Prev", width);
      b->Connect("Clicked()", cls, this, "DoPrevEvent()");

      mEventId = new TGNumberEntry(f, 0, 5, -1, TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
                                   TGNumberFormat::kNELLimitMinMax, 0, 10000);
      f->AddFrame(mEventId, new TGLayoutHints(kLHintsNormal, 10, 5, 0, 0));
      mEventId->Connect("ValueSet(Long_t)", cls, this, "DoSetEvent()");
      TGLabel* infoLabel = new TGLabel(f);
      f->AddFrame(infoLabel, new TGLayoutHints(kLHintsNormal, 5, 10, 4, 0));

      b = EventManagerFrame::makeButton(f, "Next", width);
      b->Connect("Clicked()", cls, this, "DoNextEvent()");
      b = EventManagerFrame::makeButton(f, "Last", width);
      b->Connect("Clicked()", cls, this, "DoLastEvent()");
      b = EventManagerFrame::makeButton(f, "Screenshot", 2 * width);
      b->Connect("Clicked()", cls, this, "DoScreenshot()");
    }
    SetCleanup(kDeepCleanup);
    Layout();
    MapSubwindows();
    MapWindow();
  }

  TGTextButton* EventManagerFrame::makeButton(TGCompositeFrame* p, const char* txt,
                                              Int_t width, Int_t lo, Int_t ro, Int_t to, Int_t bo)
  {
    TGTextButton* b = new TGTextButton(p, txt);

    //b->SetFont("-adobe-helvetica-bold-r-*-*-48-*-*-*-*-*-iso8859-1");

    if (width > 0) {
      b->SetWidth(width);
      b->ChangeOptions(b->GetOptions() | kFixedWidth);
    }
    p->AddFrame(b, new TGLayoutHints(kLHintsNormal, lo, ro, to, bo));
    return b;
  }

  void EventManagerFrame::DoFirstEvent()
  {
    mEventManager->GotoEvent(0);
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }

  void EventManagerFrame::DoPrevEvent()
  {
    mEventManager->PrevEvent();
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }

  void EventManagerFrame::DoNextEvent()
  {
    mEventManager->NextEvent();
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }

  void EventManagerFrame::DoLastEvent()
  {
    mEventManager->GotoEvent(-1); /// -1 means last available
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }

  void EventManagerFrame::DoSetEvent()
  {
  }

  void EventManagerFrame::DoScreenshot()
  {
  }

  void EventManagerFrame::DoTimeTick()
  {
    std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
    bool inTick;
    lck.lock();
    inTick = this->inTick;
    this->inTick = true;
    lck.unlock();
    if (inTick) {
      return;
    }

    if (mEventManager->getDataSource()->refresh()) {
      mEventManager->displayCurrentEvent();
    }
    lck.lock();
    this->inTick = false;
    lck.unlock();
  }

  void EventManagerFrame::StopTimer()
  {
    this->mTimerRunning = kFALSE;
    if (this->mTimer != nullptr) {
      this->mTimer->TurnOff();
    }
  }
  void EventManagerFrame::StartTimer()
  {
    if (this->mTimer != nullptr) {
      this->mTimer->SetTime((Long_t)(1000 * this->mTime));
      this->mTimer->Reset();
      this->mTimer->TurnOn();
    }
    this->mTimerRunning = kTRUE;
  }

  } // namespace event_visualisation
}
