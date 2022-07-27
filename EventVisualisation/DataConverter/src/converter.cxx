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
/// \file    converter.cxx
/// \author julian.myrcha@cern.ch

#include "EventVisualisationView/Initializer.h"
#include "EventVisualisationView/Options.h"
#include <EventVisualisationDataConverter/VisualisationEvent.h>
#include <EventVisualisationDataConverter/VisualisationEventJSONSerializer.h>
#include <EventVisualisationDataConverter/VisualisationEventROOTSerializer.h>
#include <TApplication.h>
#include <TEveManager.h>
#include <TEnv.h>
#include <filesystem>
#include "FairLogger.h"

int main(int argc, char** argv)
{
  LOG(info) << "Welcome in O2 event conversion tool";
  if (argc != 3) {
    LOG(error) << "two filename required, second should point to not existent file";
    exit(-1);
  }

  std::string src = argv[1];
  std::string dst = argv[2];

  o2::event_visualisation::VisualisationEvent vEvent;

  auto srcSerializer = o2::event_visualisation::VisualisationEventSerializer::getInstance(std::filesystem::path(src).extension());
  auto dstSerializer = o2::event_visualisation::VisualisationEventSerializer::getInstance(std::filesystem::path(dst).extension());

  std::chrono::time_point currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::time_point endTime = std::chrono::high_resolution_clock::now();

  srcSerializer->fromFile(vEvent, src);
  endTime = std::chrono::high_resolution_clock::now();
  LOG(info) << "read took "
            << std::chrono::duration_cast<std::chrono::microseconds>(endTime - currentTime).count() * 1e-6;

  currentTime = std::chrono::high_resolution_clock::now();
  dstSerializer->toFile(vEvent, dst);
  endTime = std::chrono::high_resolution_clock::now();
  LOG(info) << "write took "
            << std::chrono::duration_cast<std::chrono::microseconds>(endTime - currentTime).count() * 1e-6;
  return 0;
}
