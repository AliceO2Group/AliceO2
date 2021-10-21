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
/// \author m.chwasiuk@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
#define ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H

#include "EventVisualisationView/EventManager.h"
#include <TGMdiMainFrame.h>
#include <TASImage.h>

class TGTextButton;
class TGCompositeFrame;
class TGNumberEntry;
class TGLabel;

namespace o2
{
namespace event_visualisation
{

class EventManagerFrame : public TGMainFrame
{
 private:
  Float_t mTime;  // Auto-load time in seconds
  TTimer* mTimer; // Timer for automatic event loading
  bool mTimerRunning;
  bool inTick = false;
  bool setInTick();   // try set inTick, return true if set, false if already set
  void clearInTick(); // safely clears inTick
  void checkMemory(); // check memory used end exit(-1) if it is too much
  static TGTextButton* makeButton(TGCompositeFrame* p, const char* txt, Int_t width = 0,
                                  Int_t lo = 0, Int_t ro = 0, Int_t to = 0, Int_t bo = 0);
  bool CopyImage(TASImage* dst, TASImage* src, Int_t x_dst, Int_t y_dst, Int_t x_src, Int_t y_src, UInt_t w_src, UInt_t h_src);

 protected:
  o2::event_visualisation::EventManager* mEventManager; // Model object.
  TGNumberEntry* mEventId;                              // Display/edit current event id
 public:
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
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
