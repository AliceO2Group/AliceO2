// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include <TASImage.h>
#include <EventVisualisationBase/DataSourceOnline.h>
#include <EventVisualisationView/EventManagerFrame.h>
#include <EventVisualisationView/MultiView.h>
#include "EventVisualisationView/Options.h"
#include <Rtypes.h>
#include <mutex>
#include <chrono>
#include <thread>

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
      b = EventManagerFrame::makeButton(f, "Save", 2 * width);
      b->Connect("Clicked()", cls, this, "DoSave()");
      b = EventManagerFrame::makeButton(f, "Online", 2 * width);
      b->Connect("Clicked()", cls, this, "DoOnlineMode()");
      b = EventManagerFrame::makeButton(f, "Saved", 2 * width);
      b->Connect("Clicked()", cls, this, "DoSavedMode()");
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
    if (not setInTick()) {
      return;
    }
    mEventManager->GotoEvent(0);
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
    clearInTick();
  }

  void EventManagerFrame::DoPrevEvent()
  {
    if (not setInTick()) {
      return;
    }
    mEventManager->PrevEvent();
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
    clearInTick();
  }

  void EventManagerFrame::DoNextEvent()
  {
    if (not setInTick()) {
      return;
    }
    mEventManager->NextEvent();
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
    clearInTick();
  }

  void EventManagerFrame::DoLastEvent()
  {
    if (not setInTick()) {
      return;
    }
    mEventManager->GotoEvent(-1); /// -1 means last available
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
    clearInTick();
  }

  void EventManagerFrame::DoSetEvent()
  {
  }

  void EventManagerFrame::DoScreenshot()
  {
    if (not setInTick()) {
      return;
    }
    UInt_t width = 2 * 1920;
    UInt_t height = 2 * 1080;

    std::string runString = "Run:";
    std::string timestampString = "Timestamp:";
    std::string collidingsystemString = "Colliding system:";
    std::string energyString = "Energy:";

    TASImage image(width, height);

    TImage* view3dImage = MultiView::getInstance()->getView(MultiView::EViews::View3d)->GetGLViewer()->GetPictureUsingBB();
    view3dImage->Scale(width * 0.66, height);
    view3dImage->CopyArea(&image, 0, 0, view3dImage->GetWidth(), view3dImage->GetHeight(), 0, 0);

    TImage* viewRphiImage = MultiView::getInstance()->getView(MultiView::EViews::ViewRphi)->GetGLViewer()->GetPictureUsingBB();
    viewRphiImage->Scale(width * 0.33, height * 0.5);
    viewRphiImage->CopyArea(&image, 0, 0, viewRphiImage->GetWidth(), viewRphiImage->GetHeight(), width * 0.66, 0);

    TImage* viewZrhoImage = MultiView::getInstance()->getView(MultiView::EViews::ViewZrho)->GetGLViewer()->GetPictureUsingBB();
    viewZrhoImage->Scale(width * 0.33, height * 0.5);
    viewZrhoImage->CopyArea(&image, 0, 0, viewZrhoImage->GetWidth(), viewZrhoImage->GetHeight(), width * 0.66, height * 0.5);

    image.DrawText(10, 1000, runString.c_str(), 24, "#FFFFFF");
    image.DrawText(10, 1020, timestampString.c_str(), 24, "#FFFFFF");
    image.DrawText(10, 1040, collidingsystemString.c_str(), 24, "#FFFFFF");
    image.DrawText(10, 1060, energyString.c_str(), 24, "#FFFFFF");

    image.WriteImage("Screenshot.png", TImage::kPng);
    clearInTick();
  }

  void EventManagerFrame::DoTimeTick()
  {
    if (not setInTick()) {
      return;
    }
    if (mEventManager->getDataSource()->refresh()) {
      mEventManager->displayCurrentEvent();
    }
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
    clearInTick();
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

  void EventManagerFrame::DoSave()
  {
    if (!Options::Instance()->savedDataFolder().empty()) {
      if (not setInTick()) {
        return;
      }
      this->mEventManager->getDataSource()->saveCurrentEvent(Options::Instance()->savedDataFolder());
      clearInTick();
    }
  }

  void EventManagerFrame::DoOnlineMode()
  {
    if (not setInTick()) {
      return;
    }
    this->mEventManager->getDataSource()->changeDataFolder(Options::Instance()->dataFolder());
    clearInTick();
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }

  void EventManagerFrame::DoSavedMode()
  {
    if (!Options::Instance()->savedDataFolder().empty()) {
      if (not setInTick()) {
        return;
      }
      this->mEventManager->getDataSource()->changeDataFolder(Options::Instance()->savedDataFolder());
      clearInTick();
      mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
    }
  }

  bool EventManagerFrame::setInTick()
  {
    std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
    bool inTick;
    lck.lock();
    inTick = this->inTick;
    this->inTick = true;
    lck.unlock();
    return not inTick; // it is me who set inTick
  }

  void EventManagerFrame::clearInTick()
  {
    std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
    lck.lock();
    this->inTick = false;
    lck.unlock();
  }

  void EventManagerFrame::DoTerminate()
  {
    StopTimer();
    std::chrono::seconds duration(1); // wait 1 second to give a chance
    std::this_thread::sleep_for(duration);
    while (not setInTick()) { // make sure chance was taken
      continue;
    }
    exit(0);
  }

  } // namespace event_visualisation
}
