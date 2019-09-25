// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EventManagerFrame.h
/// \brief GUI (bottom buttons) for visualisation
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
#define ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H

#include "EventVisualisationBase/EventManager.h"
#include <TGMdiMainFrame.h>

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
  static TGTextButton* makeButton(TGCompositeFrame* p, const char* txt, Int_t width = 0,
                                  Int_t lo = 0, Int_t ro = 0, Int_t to = 0, Int_t bo = 0);

 protected:
  o2::event_visualisation::EventManager* mEventManager; // Model object.
  TGNumberEntry* mEventId;                              // Display/edit current event id
 public:
  EventManagerFrame(o2::event_visualisation::EventManager& eventManager);
  ~EventManagerFrame() override = default;
  ClassDefOverride(EventManagerFrame, 0); // GUI window for AliEveEventManager.

 public: // slots
  void DoFirstEvent();
  void DoPrevEvent();
  void DoNextEvent();
  void DoLastEvent();
  void DoSetEvent();
  void DoScreenshot();
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_EVENTMANAGERFRAME_H
