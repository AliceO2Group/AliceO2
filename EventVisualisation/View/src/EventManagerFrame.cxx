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
#include <TGButtonGroup.h>
#include <TGNumberEntry.h>
#include <TGLabel.h>
#include <TTimer.h>
#include <TGDoubleSlider.h>
#include <EventVisualisationBase/DataSourceOnline.h>
#include <EventVisualisationBase/ConfigurationManager.h>
#include <EventVisualisationBase/DirectoryLoader.h>
#include <EventVisualisationView/EventManagerFrame.h>
#include <EventVisualisationView/MultiView.h>
#include <EventVisualisationView/Screenshot.h>
#include <EventVisualisationView/Options.h>
#include <Rtypes.h>
#include <mutex>
#include <chrono>
#include <thread>
#include <filesystem>
#include <cassert>
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
  TGRadioButton* r = nullptr;
  TGHorizontalFrame* f = nullptr;

  auto const options = Options::Instance();

  this->mRunMode = decipherRunMode(options->dataFolder());

  f = new TGHorizontalFrame(this);
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

    b = EventManagerFrame::makeButton(f, "Next", width, "Go to the next event");
    b->Connect("Clicked()", cls, this, "DoNextEvent()");
    b = EventManagerFrame::makeButton(f, "Last", width, "Go to the last event");
    b->Connect("Clicked()", cls, this, "DoLastEvent()");
    b = EventManagerFrame::makeButton(f, "Screenshot", 2 * width, "Make a screenshot of current event");
    b->Connect("Clicked()", cls, this, "DoScreenshot()");
    b = EventManagerFrame::makeButton(f, "Save", 2 * width, "Save current event");
    b->Connect("Clicked()", cls, this, "DoSave()");

    TGHButtonGroup* g = new TGHButtonGroup(f);
    this->mOnlineModeBtn = b = EventManagerFrame::makeRadioButton(g, "Online", 2 * width, "Change data source to online events", Options::Instance()->online());
    b->Connect("Clicked()", cls, this, "DoOnlineMode()");
    this->mSavedModeBtn = b = EventManagerFrame::makeRadioButton(g, "Saved", 2 * width, "Change data source to saved events", !Options::Instance()->online());
    b->Connect("Clicked()", cls, this, "DoSavedMode()");
    this->mSequentialModeBtn = b = EventManagerFrame::makeRadioButton(g, "Sequential", 2 * width, "Sequentially display saved events", !Options::Instance()->online());
    b->Connect("Clicked()", cls, this, "DoSequentialMode()");
    f->AddFrame(g, new TGLayoutHints(kLHintsNormal, 0, 0, 0, 0));
  }

  f = new TGHorizontalFrame(this);
  {
    Int_t width = 50;
    this->AddFrame(f, new TGLayoutHints(kLHintsExpandX, 0, 0, 2, 2));

    TGLabel* infoLabel = new TGLabel(f);
    f->AddFrame(infoLabel, new TGLayoutHints(kLHintsNormal, 5, 10, 4, 0));

    f->AddFrame(infoLabel, new TGLayoutHints(kLHintsNormal, 5, 10, 4, 0));
    this->mTimeFrameSlider = EventManagerFrame::makeSlider(f, "Time", 8 * width);
    makeSliderRangeEntries(f, 30, this->mTimeFrameSliderMin, "Display the minimum value of the time",
                           this->mTimeFrameSliderMax, "Display the maximum value of the time");
    this->mTimeFrameSlider->Connect("PositionChanged()", cls, this, "DoTimeFrameSliderChanged()");

    TGHButtonGroup* g = new TGHButtonGroup(f);
    mSyntheticRunBtn = r = EventManagerFrame::makeRadioButton(g, "Synthetic", 2 * width, "Change source directory to synthetic run", Options::Instance()->online());
    r->Connect("Clicked()", cls, this, "DoSyntheticData()");
    mCosmicsRunBtn = r = EventManagerFrame::makeRadioButton(g, "Cosmics", 2 * width, "Change source directory to cosmics run", !Options::Instance()->online());
    r->Connect("Clicked()", cls, this, "DoCosmicsData()");
    mPhysicsRunBtn = r = EventManagerFrame::makeRadioButton(g, "Physics", 2 * width, "Change source directory to physics run", !Options::Instance()->online());
    r->Connect("Clicked()", cls, this, "DoPhysicsData()");
    f->AddFrame(g, new TGLayoutHints(kLHintsNormal, 0, 0, 0, 0));
  }

  this->setRunMode(mRunMode, kFALSE);
  this->mOnlineModeBtn->SetState(kButtonDown);
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

TGRadioButton* EventManagerFrame::makeRadioButton(TGButtonGroup* g, const char* txt,
                                                  Int_t width, const char* txttooltip, bool checked, Int_t lo, Int_t ro, Int_t to, Int_t bo)
{
  TGRadioButton* b = new TGRadioButton(g, txt);

  if (width > 0) {
    b->SetWidth(width);
    b->ChangeOptions(b->GetOptions() | kFixedWidth);
  }

  if (txttooltip != nullptr) {
    b->SetToolTipText(txttooltip);
  }

  b->SetOn(checked);

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
  std::error_code ec{};
  bool saveFolderExists = std::filesystem::is_directory(Options::Instance()->savedDataFolder(), ec);
  this->mSavedModeBtn->SetEnabled(saveFolderExists);
  this->mSequentialModeBtn->SetEnabled(saveFolderExists);
  this->mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  this->mTimeFrameSliderMin->SetNumber(mEventManager->getDataSource()->getTimeFrameMinTrackTime());
  this->mTimeFrameSliderMax->SetNumber(mEventManager->getDataSource()->getTimeFrameMaxTrackTime());
  switch (this->mDisplayMode) {
    case OnlineMode:
      this->mOnlineModeBtn->SetState(kButtonDown);
      break;
    case SavedMode:
      this->mSavedModeBtn->SetState(kButtonDown);
      break;
    case SequentialMode:
      this->mSequentialModeBtn->SetState(kButtonDown);
      break;
  }
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

  std::time_t time = std::time(nullptr);
  char time_str[100];
  std::strftime(time_str, sizeof(time_str), "%Y_%m_%d_%H_%M_%S", std::localtime(&time));
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);
  std::string outDirectory = settings.GetValue("screenshot.path", "Screenshots");
  std::ostringstream filepath;
  filepath << outDirectory << "/Screenshot_" << time_str << "_" << getRunTypeString(mRunMode) << ".png";
  if (!std::filesystem::is_directory(outDirectory)) {
    std::filesystem::create_directory(outDirectory);
  }
  Screenshot::perform(filepath.str(), this->mEventManager->getDataSource()->getDetectorsMask(),
                      this->mEventManager->getDataSource()->getRunNumber(),
                      this->mEventManager->getDataSource()->getFirstTForbit(),
                      this->mEventManager->getDataSource()->getCollisionTime());

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
        this->memoryUsedInfo = size;
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

void EventManagerFrame::createOutreachScreenshot()
{
  if (!Options::Instance()->imageFolder().empty()) {
    string fileName = mEventManager->getInstance().getDataSource()->getEventName();
    if (fileName.size() < 5) {
      return;
    }
    string imageFolder = Options::Instance()->imageFolder();
    if (!std::filesystem::is_directory(imageFolder)) {
      std::filesystem::create_directory(imageFolder);
    }
    fileName = imageFolder + "/" + fileName.substr(0, fileName.find_last_of('.')) + ".png";
    if (!std::filesystem::is_regular_file(fileName)) {
      std::vector<std::string> ext = {".png"};
      DirectoryLoader::removeOldestFiles(imageFolder, ext, 10);
      LOG(info) << "Outreach screenshot: " << fileName;
      Screenshot::perform(fileName, this->mEventManager->getDataSource()->getDetectorsMask(),
                          this->mEventManager->getDataSource()->getRunNumber(),
                          this->mEventManager->getDataSource()->getFirstTForbit(),
                          this->mEventManager->getDataSource()->getCollisionTime());
    }
    // LOG(info) << mEventManager->getInstance().getDataSource()->getEventName();
  }
}

void EventManagerFrame::DoTimeTick()
{
  if (not setInTick()) {
    return;
  }
  checkMemory(); // exits if memory usage too high = prevents freezing long-running machine
  this->createOutreachScreenshot();
  bool refreshNeeded = mEventManager->getDataSource()->refresh();
  if (this->mDisplayMode == SequentialMode) {
    mEventManager->getDataSource()->rollToNext();
    refreshNeeded = true;
  }

  if (refreshNeeded) {
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
  this->mEventManager->getDataSource()->changeDataFolder(getSourceDirectory(this->mRunMode).Data());
  this->mDisplayMode = OnlineMode;
  this->mEventManager->setShowDate(true);
  clearInTick();
  mEventManager->GotoEvent(-1);
  mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
}

void EventManagerFrame::DoSavedMode()
{
  if (!Options::Instance()->savedDataFolder().empty()) {
    if (not setInTick()) {
      return;
    }
    this->mEventManager->getDataSource()->changeDataFolder(getSourceDirectory(this->mRunMode).Data());
    this->mDisplayMode = SavedMode;
    this->mEventManager->setShowDate(true);
    if (mEventManager->getDataSource()->refresh()) {
      mEventManager->displayCurrentEvent();
    }
    clearInTick();
    mEventManager->GotoEvent(-1);
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }
}

void EventManagerFrame::DoSequentialMode()
{
  if (!Options::Instance()->savedDataFolder().empty()) {
    if (not setInTick()) {
      return;
    }
    this->mEventManager->getDataSource()->changeDataFolder(getSourceDirectory(this->mRunMode).Data());
    this->mDisplayMode = SequentialMode;
    this->mEventManager->setShowDate(false);
    if (mEventManager->getDataSource()->refresh()) {
      mEventManager->displayCurrentEvent();
    }
    clearInTick();
    mEventManager->GotoEvent(-1);
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }
}

void EventManagerFrame::DoSyntheticData()
{
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  if (this->mRunMode != EventManagerFrame::SyntheticRun) {
    this->mRunMode = EventManagerFrame::SyntheticRun;
    this->mEventManager->getDataSource()->changeDataFolder(getSourceDirectory(this->mRunMode).Data());

    if (mEventManager->getDataSource()->refresh()) {
      mEventManager->displayCurrentEvent();
    }
    clearInTick();
    mEventManager->GotoEvent(-1);
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }
}

void EventManagerFrame::DoCosmicsData()
{
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  if (this->mRunMode != EventManagerFrame::CosmicsRun) {
    this->mRunMode = EventManagerFrame::CosmicsRun;
    this->mEventManager->getDataSource()->changeDataFolder(getSourceDirectory(this->mRunMode).Data());

    if (mEventManager->getDataSource()->refresh()) {
      mEventManager->displayCurrentEvent();
    }
    clearInTick();
    mEventManager->GotoEvent(-1);
    mEventId->SetIntNumber(mEventManager->getDataSource()->getCurrentEvent());
  }
}

void EventManagerFrame::DoPhysicsData()
{
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  if (this->mRunMode != EventManagerFrame::PhysicsRun) {
    this->mRunMode = EventManagerFrame::PhysicsRun;
    this->mEventManager->getDataSource()->changeDataFolder(getSourceDirectory(this->mRunMode).Data());

    if (mEventManager->getDataSource()->refresh()) {
      mEventManager->displayCurrentEvent();
    }
    clearInTick();
    mEventManager->GotoEvent(-1);
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

float EventManagerFrame::getMinTimeFrameSliderValue() const
{
  return mTimeFrameSlider->GetMinPosition();
}

float EventManagerFrame::getMaxTimeFrameSliderValue() const
{
  return mTimeFrameSlider->GetMaxPosition();
}

void EventManagerFrame::setRunMode(EventManagerFrame::RunMode runMode, Bool_t emit)
{
  if (emit) {
    this->mEventManager->getDataSource()->changeDataFolder(getSourceDirectory(this->mRunMode).Data());
  } else {
    mSyntheticRunBtn->SetState(EButtonState::kButtonUp, kFALSE);
    mCosmicsRunBtn->SetState(EButtonState::kButtonUp, kFALSE);
    mPhysicsRunBtn->SetState(EButtonState::kButtonUp, kFALSE);
  }

  switch (runMode) {
    case EventManagerFrame::SyntheticRun:
      this->mRunMode = CosmicsRun;
      mSyntheticRunBtn->SetState(EButtonState::kButtonDown, emit);
      break;
    case EventManagerFrame::CosmicsRun:
      this->mRunMode = PhysicsRun;
      mCosmicsRunBtn->SetState(EButtonState::kButtonDown, emit);
      break;
    case EventManagerFrame::PhysicsRun:
      this->mRunMode = SyntheticRun;
      mPhysicsRunBtn->SetState(EButtonState::kButtonDown, emit);
      break;
  }
}

TString EventManagerFrame::getSourceDirectory(EventManagerFrame::RunMode runMode)
{
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);

  switch (runMode) {
    case EventManagerFrame::SyntheticRun:
      return settings.GetValue("data.synthetic.run.dir", "jsons/synthetic");
    case EventManagerFrame::CosmicsRun:
      return settings.GetValue("data.cosmics.run.dir", "jsons/cosmics");
    case EventManagerFrame::PhysicsRun:
      return settings.GetValue("data.physics.run.dir", "jsons/physics");
    default:
      return settings.GetValue("data.synthetic.run.dir", "jsons/synthetic");
  }
}

EventManagerFrame::RunMode EventManagerFrame::decipherRunMode(TString name, RunMode defaultRun)
{
  if (name == "SYNTHETIC") {
    return SyntheticRun;
  } else if (name == "COSMICS") {
    return CosmicsRun;
  } else if (name == "PHYSICS") {
    return PhysicsRun;
  } else {
    return defaultRun;
  }
}

TString EventManagerFrame::getRunTypeString(EventManagerFrame::RunMode runMode)
{
  switch (runMode) {
    case EventManagerFrame::SyntheticRun:
      return "synthetic";
    case EventManagerFrame::CosmicsRun:
      return "cosmics";
    case EventManagerFrame::PhysicsRun:
      return "physics";
    default:
      return "";
  }
}

} // namespace event_visualisation
} // namespace o2
