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

class DataInterpreterITS final : public DataInterpreter
{
 public:
  // Default constructor
  DataInterpreterITS();

  // Default destructor
  ~DataInterpreterITS() final = default;

  // Returns a visualisation Event for this data type
  VisualisationEvent interpretDataForType(TObject* data, EVisualisationDataType type) final;
};

} // namespace event_visualisation
} // namespace o2

#endif //ALICE_O2_EVENTVISUALISATION_BASE_DATAINTERPRETERITS_H
