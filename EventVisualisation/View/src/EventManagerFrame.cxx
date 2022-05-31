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
#include <TGDoubleSlider.h>
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
#include <cassert>
#include <fstream>
#include <FairLogger.h>

std::mutex mtx; // mutex for critical section

ClassImp(o2::event_visualisation::EventManagerFrame);

namespace o2
{
namespace event_visualisation
{

EventManagerFrame* EventManagerFrame::mInstance = nullptr;
EventManagerFrame& EventManagerFrame::getInstance()
{
  assert(mInstance != nullptr);
  return *mInstance;
}

EventManagerFrame::~EventManagerFrame()
{
  this->StopTimer();
}

EventManagerFrame::EventManagerFrame(o2::event_visualisation::EventManager& eventManager)
  : TGMainFrame(gClient->GetRoot(), 400, 100, kVerticalFrame)
{
  mInstance = this;
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

    b = EventManagerFrame::makeButton(f, "First", width, "Go to the first event");
    b->Connect("Clicked()", cls, this, "DoFirstEvent()");
    b = EventManagerFrame::makeButton(f, "Prev", width, "Go to the previous event");
    b->Connect("Clicked()", cls, this, "DoPrevEvent()");

    mEventId = new TGNumberEntry(f, 0, 5, -1, TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
                                 TGNumberFormat::kNELLimitMinMax, 0, 10000);
    f->AddFrame(mEventId, new TGLayoutHints(kLHintsNormal, 10, 5, 0, 0));
    mEventId->Connect("ValueSet(Long_t)", cls, this, "DoSetEvent()");
    TGLabel* infoLabel = new TGLabel(f);
    f->AddFrame(infoLabel, new TGLayoutHints(kLHintsNormal, 5, 10, 4, 0));

    b = EventManagerFrame::makeButton(f, "Next", width, "Go to the next event");
    b->Connect("Clicked()", cls, this, "DoNextEvent()");
    b = EventManagerFrame::makeButton(f, "Last", width, "Go to the last event");
    b->Connect("Clicked()", cls, this, "DoLastEvent()");
    b = EventManagerFrame::makeButton(f, "Screenshot", 2 * width, "Make a screenshot of current event");
    b->Connect("Clicked()", cls, this, "DoScreenshot()");
    b = EventManagerFrame::makeButton(f, "Save", 2 * width, "Save current event");
    b->Connect("Clicked()", cls, this, "DoSave()");
    b = EventManagerFrame::makeButton(f, "Online", 2 * width, "Change data source to online events");
    b->Connect("Clicked()", cls, this, "DoOnlineMode()");
    b = EventManagerFrame::makeButton(f, "Saved", 2 * width, "Change data source to saved events");
    b->Connect("Clicked()", cls, this, "DoSavedMode()");

    f->AddFrame(infoLabel, new TGLayoutHints(kLHintsNormal, 5, 10, 4, 0));
    this->mTimeFrameSlider = EventManagerFrame::makeSlider(f, "Time", 8 * width);
    makeSliderRangeEntries(f, 30, this->mTimeFrameSliderMin, "Display the minimum value of the time",
                           this->mTimeFrameSliderMax, "Display the maximum value of the time");
    this->mTimeFrameSlider->Connect("PositionChanged()", cls, this, "DoTimeFrameSliderChanged()");
  }
  SetCleanup(kDeepCleanup);
  Layout();
  MapSubwindows();
  MapWindow();
}

TGTextButton* EventManagerFrame::makeButton(TGCompositeFrame* p, const char* txt,
                                            Int_t width, const char* txttooltip, Int_t lo, Int_t ro, Int_t to, Int_t bo)
{
  TGTextButton* b = new TGTextButton(p, txt);

  if (width > 0) {
    b->SetWidth(width);
    b->ChangeOptions(b->GetOptions() | kFixedWidth);
  }

  if (txttooltip != nullptr) {
    b->SetToolTipText(txttooltip);
  }

  p->AddFrame(b, new TGLayoutHints(kLHintsNormal, lo, ro, to, bo));
  return b;
}

TGDoubleHSlider* EventManagerFrame::makeSlider(TGCompositeFrame* p, const char* txt, Int_t width,
                                               Int_t lo, Int_t ro, Int_t to, Int_t bo)
{
  TGCompositeFrame* sliderFrame = new TGCompositeFrame(p, width, 20, kHorizontalFrame);
  TGLabel* sliderLabel = new TGLabel(sliderFrame, txt);
  sliderFrame->AddFrame(sliderLabel,
                        new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));
  TGDoubleHSlider* slider = new TGDoubleHSlider(sliderFrame, width - 80, kDoubleScaleBoth);
  slider->SetRange(0, MaxRange);
  slider->SetPosition(0, MaxRange);
  sliderFrame->AddFrame(slider, new TGLayoutHints(kLHintsLeft));
  p->AddFrame(sliderFrame, new TGLayoutHints(kLHintsTop, lo, ro, to, bo));
  return slider;
}

void EventManagerFrame::makeSliderRangeEntries(TGCompositeFrame* parent, int height,
                                               TGNumberEntryField*& minEntry, const TString& minToolTip,
                                               TGNumberEntryField*& maxEntry, const TString& maxToolTip)
{
  TGCompositeFrame* frame = new TGCompositeFrame(parent, 80, height, kHorizontalFrame);

  minEntry = new TGNumberEntryField(frame, -1, 0., TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEAAnyNumber);
  minEntry->SetToolTipText(minToolTip.Data());
  minEntry->Resize(100, height);
  minEntry->SetState(false);
  frame->AddFrame(minEntry, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));

  maxEntry = new TGNumberEntryField(frame, -1, 0., TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEAAnyNumber);
  maxEntry->SetToolTipText(maxToolTip.Data());
  maxEntry->Resize(100, height);
  maxEntry->SetState(false);
  frame->AddFrame(maxEntry, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
  parent->AddFrame(frame, new TGLayoutHints(kLHintsTop, 5, 0, 0, 0));
}

void EventManagerFrame::updateGUI()
{
  this->mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  this->mTimeFrameSliderMin->SetNumber(mEventManager->getDataSource()->getTimeFrameMinTrackTime());
  this->mTimeFrameSliderMax->SetNumber(mEventManager->getDataSource()->getTimeFrameMaxTrackTime());
}

void EventManagerFrame::DoTimeFrameSliderChanged()
{
  if (not setInTick()) {
    return;
  }
  this->mEventManager->CurrentEvent();
  this->updateGUI();
  clearInTick();
}

void EventManagerFrame::DoFirstEvent()
{
  if (not setInTick()) {
    return;
  }
  mEventManager->GotoEvent(0);
  this->updateGUI();
  clearInTick();
}

void EventManagerFrame::DoPrevEvent()
{
  if (not setInTick()) {
    return;
  }
  mEventManager->PrevEvent();
  this->updateGUI();
  clearInTick();
}

void EventManagerFrame::DoNextEvent()
{
  if (not setInTick()) {
    return;
  }
  mEventManager->NextEvent();
  this->updateGUI();
  clearInTick();
}

void EventManagerFrame::DoLastEvent()
{
  if (not setInTick()) {
    return;
  }
  mEventManager->GotoEvent(-1); /// -1 means last available
  this->updateGUI();
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
  const char* backgroundColor = "#000000"; // "#19324b";
  const char* outDirectory = "Screenshots";

  std::string runString = "Run:";
  std::string timestampString = "Timestamp:";
  std::string collidingsystemString = "Colliding system:";
  std::string energyString = "Energy:";

  std::time_t time = std::time(nullptr);
  char time_str[100];
  std::strftime(time_str, sizeof(time_str), "%Y_%m_%d_%H_%M_%S", std::localtime(&time));

  TASImage* scaledImage;

  std::ostringstream filepath;
  filepath << outDirectory << "/Screenshot_" << time_str << ".png";

  TASImage image(width, height);
  image.FillRectangle(backgroundColor, 0, 0, width, height);

  const auto annotationStateTop = MultiView::getInstance()->getAnnotationTop()->GetState();
  const auto annotationStateBottom = MultiView::getInstance()->getAnnotationBottom()->GetState();
  MultiView::getInstance()->getAnnotationTop()->SetState(TGLOverlayElement::kInvisible);
  MultiView::getInstance()->getAnnotationBottom()->SetState(TGLOverlayElement::kInvisible);

  TImage* view3dImage = MultiView::getInstance()->getView(MultiView::EViews::View3d)->GetGLViewer()->GetPictureUsingBB();

  MultiView::getInstance()->getAnnotationTop()->SetState(annotationStateTop);
  MultiView::getInstance()->getAnnotationBottom()->SetState(annotationStateBottom);

  scaledImage = ScaleImage((TASImage*)view3dImage, width * 0.65, height * 0.95);
  if (scaledImage) {
    CopyImage(&image, scaledImage, width * 0.015, height * 0.025, 0, 0, scaledImage->GetWidth(), scaledImage->GetHeight());
    delete scaledImage;
  }

  TImage* viewRphiImage = MultiView::getInstance()->getView(MultiView::EViews::ViewRphi)->GetGLViewer()->GetPictureUsingBB();
  scaledImage = ScaleImage((TASImage*)viewRphiImage, width * 0.3, height * 0.45);
  if (scaledImage) {
    CopyImage(&image, scaledImage, width * 0.68, height * 0.025, 0, 0, scaledImage->GetWidth(), scaledImage->GetHeight());
    delete scaledImage;
  }

  TImage* viewZrhoImage = MultiView::getInstance()->getView(MultiView::EViews::ViewZrho)->GetGLViewer()->GetPictureUsingBB();
  scaledImage = ScaleImage((TASImage*)viewZrhoImage, width * 0.3, height * 0.45);
  if (scaledImage) {
    CopyImage(&image, scaledImage, width * 0.68, height * 0.525, 0, 0, scaledImage->GetWidth(), scaledImage->GetHeight());
    delete scaledImage;
  }

  bool logo = true;
  if (logo) {
    TASImage* aliceLogo = new TASImage("Alice.png");
    if (aliceLogo->IsValid()) {
      double ratio = 1434. / 1939.;
      aliceLogo->Scale(0.08 * width, 0.08 * width / ratio);
      image.Merge(aliceLogo, "alphablend", 20, 20);
      delete aliceLogo;
    }
  }

  int fontSize = 0.015 * height;
  int textX;
  int textLineHeight = 0.015 * height;
  int textY;

  if (logo) {
    TASImage* o2Logo = new TASImage("o2.png");
    if (o2Logo->IsValid()) {
      double ratio = (double)(o2Logo->GetWidth()) / (double)(o2Logo->GetHeight());
      int o2LogoX = 0.01 * width;
      int o2LogoY = 0.01 * width;
      int o2LogoSize = 0.04 * width;
      o2Logo->Scale(o2LogoSize, o2LogoSize / ratio);
      image.Merge(o2Logo, "alphablend", o2LogoX, height - o2LogoSize / ratio - o2LogoY);
      textX = o2LogoX + o2LogoSize + o2LogoX;
      textY = height - o2LogoSize / ratio - o2LogoY;
      delete o2Logo;
    } else {
      textX = 229;
      textY = 1926;
    }
  }

  o2::dataformats::GlobalTrackID::mask_t detectorsMask;

  auto detectorsString = dataformats::GlobalTrackID::getSourcesNames(this->mEventManager->getDataSource()->getDetectorsMask());

  std::vector<std::string> lines;
  std::ifstream input("screenshot.txt");
  if (input.is_open()) {
    for (std::string line; getline(input, line);) {
      lines.push_back(line);
    }
  }

  if (!this->mEventManager->getDataSource()->getCollisionTime().empty()) {
    lines.push_back((std::string)TString::Format("Run number: %d", this->mEventManager->getDataSource()->getRunNumber()));
    lines.push_back((std::string)TString::Format("First TF orbit: %d", this->mEventManager->getDataSource()->getFirstTForbit()));
    lines.push_back((std::string)TString::Format("Date: %s", this->mEventManager->getDataSource()->getCollisionTime().c_str()));
    lines.push_back((std::string)TString::Format("Detectors: %s", detectorsString.c_str()));
  }

  image.BeginPaint();

  for (int i = 0; i < 4; i++) {
    image.DrawText(textX, textY + i * textLineHeight, lines[i].c_str(), fontSize, "#BBBBBB", "FreeSansBold.otf");
  }
  image.EndPaint();

  if (!std::filesystem::is_directory(outDirectory)) {
    std::filesystem::create_directory(outDirectory);
  }
  image.WriteImage(filepath.str().c_str(), TImage::kPng);
  clearInTick();
}

void EventManagerFrame::checkMemory()
{
  const long memoryLimit = Options::Instance()->memoryLimit();
  if (memoryLimit != -1) {
    const char* statmPath = "/proc/self/statm";
    long size = -1;
    FILE* f = fopen(statmPath, "r");
    if (f != nullptr) { // could not read file => no check
      int success = fscanf(f, "%ld", &size);
      fclose(f);
      if (success == 1) {       // properly readed
        size = 4 * size / 1024; // in MB
        LOG(info) << "Memory used: " << size << " memory allowed: " << memoryLimit;
        if (size > memoryLimit) {
          LOG(error) << "Memory used: " << size << " exceeds memory allowed: "
                     << memoryLimit;
          exit(-1);
        }
      }
    }
  }
}

void EventManagerFrame::DoTimeTick()
{
  if (not setInTick()) {
    return;
  }
  checkMemory(); // exits if memory usage too high = prevents freezing long-running machine
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

TASImage* EventManagerFrame::ScaleImage(TASImage* image, UInt_t desiredWidth, UInt_t desiredHeight)
{
  if (!image) {
    return nullptr;
  }
  if (desiredWidth == 0 || desiredHeight == 0) {
    return nullptr;
  }

  const char* backgroundColor = "#000000";

  UInt_t scaleWidth = desiredWidth;
  UInt_t scaleHeight = desiredHeight;
  UInt_t offsetWidth = 0;
  UInt_t offsetHeight = 0;

  float aspectRatio = (float)image->GetWidth() / (float)image->GetHeight();

  if (desiredWidth >= aspectRatio * desiredHeight) {
    scaleWidth = (UInt_t)(aspectRatio * desiredHeight);
    offsetWidth = (desiredWidth - scaleWidth) / 2.0f;
  } else {
    scaleHeight = (UInt_t)((1.0f / aspectRatio) * desiredWidth);
    offsetHeight = (desiredHeight - scaleHeight) / 2.0f;
  }

  TASImage* scaledImage = new TASImage(desiredWidth, desiredHeight);
  scaledImage->FillRectangle(backgroundColor, 0, 0, desiredWidth, desiredHeight);

  image->Scale(scaleWidth, scaleHeight);

  CopyImage(scaledImage, image, offsetWidth, offsetHeight, 0, 0, scaleWidth, scaleHeight);

  return scaledImage;
}

float EventManagerFrame::getMinTimeFrameSliderValue() const
{
  return mTimeFrameSlider->GetMinPosition();
}

float EventManagerFrame::getMaxTimeFrameSliderValue() const
{
  return mTimeFrameSlider->GetMaxPosition();
}

} // namespace event_visualisation
} // namespace o2
