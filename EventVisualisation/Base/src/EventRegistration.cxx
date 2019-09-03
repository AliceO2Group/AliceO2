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
/// \file   EventRegistration.cxx
/// \brief  breaking link dependency between EventVisualisation modules (here MultiView can register)
/// \author julian.myrcha@cern.ch

#include <EventVisualisationBase/EventRegistration.h>

namespace o2
{
namespace event_visualisation
{

EventRegistration* EventRegistration::instance = nullptr;
}
} // namespace o2
