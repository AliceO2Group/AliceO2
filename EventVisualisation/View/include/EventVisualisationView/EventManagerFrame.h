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

/// \file EventManagerFrame.h
/// \brief GUI (bottom buttons) for visualisation
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch
/// \author michal.chwesiuk@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
#define ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H

#include "EventVisualisationView/EventManager.h"
#include <TGMdiMainFrame.h>
#include <TASImage.h>

class TGTextButton;
class TGRadioButton;
class TGButtonGroup;
class TGCompositeFrame;
class TGNumberEntry;
class TGLabel;
class TGNumberEntryField;
class TGDoubleHSlider;

namespace o2
{
namespace event_visualisation
{

class EventManagerFrame : public TGMainFrame
{
 private:
  static EventManagerFrame* mInstance;     // Instance
  TGDoubleHSlider* mTimeFrameSlider;       // Slider to narrow TimeFrame data
  TGNumberEntryField* mTimeFrameSliderMin; // Number entry for slider's min.
  TGNumberEntryField* mTimeFrameSliderMax; // Number entry for slider's max.

  Float_t mTime;  // Auto-load time in seconds
  TTimer* mTimer; // Timer for automatic event loading
  bool mTimerRunning;
  bool inTick = false;
  bool setInTick();   // try set inTick, return true if set, false if already set
  void clearInTick(); // safely clears inTick
  void checkMemory(); // check memory used end exit(-1) if it is too much
  void updateGUI();   // updates
  static TGTextButton* makeButton(TGCompositeFrame* p, const char* txt, Int_t width = 0, const char* txttooltip = nullptr,
                                  Int_t lo = 0, Int_t ro = 0, Int_t to = 0, Int_t bo = 0);
  static TGRadioButton* makeRadioButton(TGButtonGroup* g, const char* txt, Int_t width = 0, const char* txttooltip = nullptr, bool checked = false,
                                        Int_t lo = 0, Int_t ro = 0, Int_t to = 0, Int_t bo = 0);
  static TGDoubleHSlider* makeSlider(TGCompositeFrame* p, const char* txt, Int_t width = 0,
                                     Int_t lo = 2, Int_t ro = 2, Int_t to = 2, Int_t bo = 2);
  static void makeSliderRangeEntries(TGCompositeFrame* parent, int height,
                                     TGNumberEntryField*& minEntry, const TString& minToolTip,
                                     TGNumberEntryField*& maxEntry, const TString& maxToolTip);

  bool CopyImage(TASImage* dst, TASImage* src, Int_t x_dst = 0, Int_t y_dst = 0, Int_t x_src = 0, Int_t y_src = 0, UInt_t w_src = 0, UInt_t h_src = 0);
  TASImage* ScaleImage(TASImage* image, UInt_t desiredWidth, UInt_t desiredHeight);

 protected:
  o2::event_visualisation::EventManager* mEventManager; // Model object.
  TGNumberEntry* mEventId;                              // Display/edit current event id
 public:
  /// Returns an instance of EventManagerFrame
  static EventManagerFrame& getInstance();
  enum ERange {
    MaxRange = 100
  };
  float getMinTimeFrameSliderValue() const;
  float getMaxTimeFrameSliderValue() const;

  EventManagerFrame(o2::event_visualisation::EventManager& eventManager);
  ~EventManagerFrame() override;
  ClassDefOverride(EventManagerFrame, 0); // GUI window for AliEveEventManager.

 public: // slots
  void DoFirstEvent();
  void DoPrevEvent();
  void DoNextEvent();
  void DoLastEvent();
  void DoSetEvent();
  void DoScreenshot();
  void DoSave();
  void DoOnlineMode();
  void DoSavedMode();
  void DoTimeTick();
  void DoTerminate();
  void StopTimer();
  void StartTimer();
  void DoTimeFrameSliderChanged();
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
