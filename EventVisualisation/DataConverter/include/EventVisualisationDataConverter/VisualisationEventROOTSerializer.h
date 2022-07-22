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

#ifndef O2EVE_VISUALISATIONEVENTROOTSERIALIZER_H
#define O2EVE_VISUALISATIONEVENTROOTSERIALIZER_H

#include "EventVisualisationDataConverter/VisualisationEventSerializer.h"
#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include <string>
#include <TFile.h>

namespace o2
{
namespace event_visualisation
{

class VisualisationEventROOTSerializer : public VisualisationEventSerializer
{
  static void save(const char* name, int value);
  static int readInt(TFile& f, const char* name);
  static void save(const char* name, const std::string& value);
  static std::string readString(TFile& f, const char* name);

 public:
  bool fromFile(VisualisationEvent& event, std::string fileName) override;
  void toFile(const VisualisationEvent& event, std::string fileName) override;
  ~VisualisationEventROOTSerializer() override = default;
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_VISUALISATIONEVENTROOTSERIALIZER_H
