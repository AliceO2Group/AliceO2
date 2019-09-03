// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   EventRegistration.h
/// \brief  breaking link dependency between EventVisualisation modules (here MultiView can register)
/// \author julian.myrcha@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_EVENTREGISTRATION_H
#define ALICE_O2_EVENTVISUALISATION_BASE_EVENTREGISTRATION_H

#include <TEveElement.h>

namespace o2
{
namespace event_visualisation
{

class EventRegistration
{
 private:
  static EventRegistration* instance;

 public:
  /// Registers an element to be drawn
  virtual void registerElement(TEveElement* event) = 0;

  /// Removes all shapes representing current event
  virtual void destroyAllEvents() = 0;

  static EventRegistration* getInstance() { return instance; }
  static void setInstance(EventRegistration* instance) { EventRegistration::instance = instance; }
};

} // namespace event_visualisation
} // namespace o2
#endif //ALICE_O2_EVENTVISUALISATION_BASE_EVENTREGISTRATION_H
