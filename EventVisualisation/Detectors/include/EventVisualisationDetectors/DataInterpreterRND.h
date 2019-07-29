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
/// \file    DataInterpreterRND.h
/// \author  Jeremi Niedziela

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERRND_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERRND_H

#include "EventVisualisationBase/DataInterpreter.h"
#include "EventVisualisationBase/EventManager.h"
#include "EventVisualisationBase/VisualisationConstants.h"

namespace o2  {
namespace event_visualisation {
  
/// DataInterpreterRND prepares random events
///
/// This class overrides DataInterpreter and implements method
/// returning visualisation objects representing random event
/// with tracks colored by PID only.
  
class DataInterpreterRND : public DataInterpreter
{
public:
  // Default constructor
  DataInterpreterRND();
  // Default destructor
  ~DataInterpreterRND() final;
  
  // Returns a list of random tracks colored by PID
  TEveElement* interpretDataForType(TObject* data, EVisualisationDataType type) final;
};
  
}
}

#endif
