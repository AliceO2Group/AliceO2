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
/// \file    VisualisationEventSerializer.h
/// \author  Julian Myrcha
///
#ifndef O2EVE_VISUALISATIONEVENTSERIALIZER_H
#define O2EVE_VISUALISATIONEVENTSERIALIZER_H

#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include <string>

namespace o2
{
namespace event_visualisation
{

class VisualisationEventSerializer
{
  static VisualisationEventSerializer* instance;

 protected:
  VisualisationEventSerializer() = default;
  static std::string fileNameIndexed(const std::string fileName, const int index);

 public:
  static VisualisationEventSerializer* getInstance() { return instance; }
  static void setInstance(VisualisationEventSerializer* newInstance)
  { // take ownership
    delete instance;
    instance = newInstance;
  }
  virtual bool fromFile(VisualisationEvent& event, std::string fileName) = 0;
  virtual void toFile(const VisualisationEvent& event, std::string fileName) = 0;
  virtual ~VisualisationEventSerializer() = default;
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_VISUALISATIONEVENTSERIALIZER_H
