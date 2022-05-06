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

///
/// \file   VisualisationEventSerializer.cxx
/// \brief  Serialization VisualisationEvent
/// \author julian.myrcha@cern.ch

#include "EventVisualisationDataConverter/VisualisationEventSerializer.h"
#include "EventVisualisationDataConverter/VisualisationEventJSONSerializer.h"
#include "FairLogger.h"
#include <iostream>
#include <iomanip>

namespace o2
{
namespace event_visualisation
{

VisualisationEventSerializer* VisualisationEventSerializer::instance = new VisualisationEventJSONSerializer();

std::string VisualisationEventSerializer::fileNameIndexed(const std::string fileName, const int index)
{
  std::stringstream buffer;
  buffer << fileName << std::setfill('0') << std::setw(3) << index << ".json";
  return buffer.str();
}

} // namespace event_visualisation
} // namespace o2