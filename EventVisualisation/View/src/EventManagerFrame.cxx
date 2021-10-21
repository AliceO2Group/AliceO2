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
#include <filesystem>

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
    UInt_t width = 3840;
    UInt_t height = 2160;
    UInt_t font_size = 30;
    UInt_t text_leading = 40;
    const char* fontColor = "#FFFFFF";
    const char* backgroundColor = "#19324b";
    const char* outDirectory = "Screenshots";

    std::string runString = "Run:";
    std::string timestampString = "Timestamp:";
    std::string collidingsystemString = "Colliding system:";
    std::string energyString = "Energy:";

    std::time_t time = std::time(nullptr);
    char time_str[100];
    std::strftime(time_str, sizeof(time_str), "%Y_%m_%d_%H_%M_%S", std::localtime(&time));

    std::ostringstream filepath;
    filepath << outDirectory << "/Screenshot_" << time_str << ".png";

    TASImage image(width, height);

    image.FillRectangle(backgroundColor, 0, 0, width, height);

    TImage* view3dImage = MultiView::getInstance()->getView(MultiView::EViews::View3d)->GetGLViewer()->GetPictureUsingBB();
    view3dImage->Scale(width * 0.65, height * 0.95);
    CopyImage(&image, (TASImage*)view3dImage, width * 0.015, height * 0.025, 0, 0, view3dImage->GetWidth(), view3dImage->GetHeight());

    TImage* viewRphiImage = MultiView::getInstance()->getView(MultiView::EViews::ViewRphi)->GetGLViewer()->GetPictureUsingBB();
    viewRphiImage->Scale(width * 0.3, height * 0.45);
    CopyImage(&image, (TASImage*)viewRphiImage, width * 0.68, height * 0.025, 0, 0, viewRphiImage->GetWidth(), viewRphiImage->GetHeight());

    TImage* viewZrhoImage = MultiView::getInstance()->getView(MultiView::EViews::ViewZrho)->GetGLViewer()->GetPictureUsingBB();
    viewZrhoImage->Scale(width * 0.3, height * 0.45);
    CopyImage(&image, (TASImage*)viewZrhoImage, width * 0.68, height * 0.525, 0, 0, viewZrhoImage->GetWidth(), viewZrhoImage->GetHeight());

    image.DrawText(10, height - 4 * text_leading, runString.c_str(), font_size, fontColor);
    image.DrawText(10, height - 3 * text_leading, timestampString.c_str(), font_size, fontColor);
    image.DrawText(10, height - 2 * text_leading, collidingsystemString.c_str(), font_size, fontColor);
    image.DrawText(10, height - 1 * text_leading, energyString.c_str(), font_size, fontColor);

    if (!std::filesystem::is_directory(outDirectory)) {
      std::filesystem::create_directory(outDirectory);
    }
    image.WriteImage(filepath.str().c_str(), TImage::kPng);

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

  bool EventManagerFrame::CopyImage(TASImage* dst, TASImage* src, Int_t x_dst, Int_t y_dst, Int_t x_src, Int_t y_src,
                                    UInt_t w_src, UInt_t h_src)
  {

    if (!dst) {
      return false;
    }
    if (!src) {
      return false;
    }

    int x = 0;
    int y = 0;
    int idx_src = 0;
    int idx_dst = 0;
    x_src = x_src < 0 ? 0 : x_src;
    y_src = y_src < 0 ? 0 : y_src;

    if ((x_src >= (int)src->GetWidth()) || (y_src >= (int)src->GetHeight())) {
      return false;
    }

    w_src = x_src + w_src > src->GetWidth() ? src->GetWidth() - x_src : w_src;
    h_src = y_src + h_src > src->GetHeight() ? src->GetHeight() - y_src : h_src;
    UInt_t yy = (y_src + y) * src->GetWidth();

    src->BeginPaint(false);
    dst->BeginPaint(false);

    UInt_t* dst_image_array = dst->GetArgbArray();
    UInt_t* src_image_array = src->GetArgbArray();

    if (!dst_image_array || !src_image_array) {
      return false;
    }

    for (y = 0; y < (int)h_src; y++) {
      for (x = 0; x < (int)w_src; x++) {

        idx_src = yy + x + x_src;
        idx_dst = (y_dst + y) * dst->GetWidth() + x + x_dst;

        if ((x + x_dst < 0) || (y_dst + y < 0) ||
            (x + x_dst >= (int)dst->GetWidth()) || (y + y_dst >= (int)dst->GetHeight())) {
          continue;
        }

        dst_image_array[idx_dst] = src_image_array[idx_src];
      }
      yy += src->GetWidth();
    }

    return true;
  }

  } // namespace event_visualisation
}
