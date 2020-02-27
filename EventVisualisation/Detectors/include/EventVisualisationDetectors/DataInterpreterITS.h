// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataInterpreterITS.h
/// \brief converting ITS data to Event Visualisation primitives
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_DETECTORS_DATAINTERPRETERITS_H
#define ALICE_O2_EVENTVISUALISATION_DETECTORS_DATAINTERPRETERITS_H

///
/// This class overrides DataInterpreter and implements method
/// returning visualisation objects representing data from ITS file.

#include "EventVisualisationBase/DataInterpreter.h"
#include "EventVisualisationBase/VisualisationConstants.h"

namespace o2
{
namespace event_visualisation
{

class DataInterpreterITS : public DataInterpreter
{
 public:
  // Default constructor
  DataInterpreterITS();

  // Default destructor
  ~DataInterpreterITS() final = default;

  // Returns a visualisation Event for this data type
  std::unique_ptr<VisualisationEvent> interpretDataForType(TObject* data, EVisualisationDataType type) final;
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERITS_H
