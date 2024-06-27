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

#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include "EventVisualisationView/Initializer.h"
#include "EventVisualisationView/Options.h"
#include <EventVisualisationDataConverter/VisualisationEventJSONSerializer.h>
#include <EventVisualisationDataConverter/VisualisationEventROOTSerializer.h>
#include <EventVisualisationDataConverter/VisualisationEventOpenGLSerializer.h>
#include <EventVisualisationBase/DirectoryLoader.h>
#include <TApplication.h>
#include <TEveManager.h>
#include <TEnv.h>
#include <filesystem>
#include <fairlogger/Logger.h>
#include <csignal>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

// source file name, destination (not existing) file name, if limit > 0 then limit EACH type of data
int singleFileConversion(const std::string& src, const std::string& dst, const int limit = -1)
{
  LOGF(info, "Translate: %s -> %s", src, dst);
  o2::event_visualisation::VisualisationEvent vEvent;

  auto srcSerializer = o2::event_visualisation::VisualisationEventSerializer::getInstance(
    std::filesystem::path(src).extension());
  auto dstSerializer = o2::event_visualisation::VisualisationEventSerializer::getInstance(
    std::filesystem::path(dst).extension());

  std::chrono::time_point currentTime = std::chrono::high_resolution_clock::now();
  std::chrono::time_point endTime = std::chrono::high_resolution_clock::now();

  srcSerializer->fromFile(vEvent, src);
  endTime = std::chrono::high_resolution_clock::now();
  // LOGF(info, "read took %f", std::chrono::duration_cast<std::chrono::microseconds>(endTime - currentTime).count() * 1e-6);
  if (limit > 0) {
    vEvent = vEvent.limit(limit);
  }

  currentTime = std::chrono::high_resolution_clock::now();
  dstSerializer->toFile(vEvent, dst);
  endTime = std::chrono::high_resolution_clock::now();
  // LOGF(info, "write took %f", std::chrono::duration_cast<std::chrono::microseconds>(endTime - currentTime).count() * 1e-6);
  return 0;
}

// reads source folder files, find missing files in destination folder and convert them
// source folder (/path-to-folder/.ext1) , destination folder (/path-to-folder/.ext2)
int folderConversion(const std::string& srcFolder, const std::string& dstFolder)
{
  std::vector<std::string> supported = {".json", ".root", ".eve"};
  auto ext1 = srcFolder.substr(srcFolder.rfind('.'));
  auto ext2 = dstFolder.substr(dstFolder.rfind('.'));

  if (supported.end() == std::find(supported.begin(), supported.end(), ext1)) {
    LOGF(error, "source folder should end with source extension:  /path-to-folder/.ext1 ");
    exit(-1);
  }
  if (supported.end() == std::find(supported.begin(), supported.end(), ext2)) {
    LOGF(error, "destination folder should end with destination extension:  /path-to-folder/.ext2 ");
    return -1;
  }
  auto src = srcFolder.substr(0, srcFolder.size() - std::string(ext1).size());
  auto dst = dstFolder.substr(0, dstFolder.size() - std::string(ext2).size());

  if (src == dst) {
    LOGF(error, "source folder same as destination folder ");
    return -1;
  }
  if (!std::filesystem::is_directory(src)) {
    LOGF(error, "source folder do not exist ");
    return -1;
  }
  if (!std::filesystem::is_directory(dst)) {
    LOGF(error, "destination folder do not exist ");
    return -1;
  }
  std::vector<std::string> vExt1 = {ext1};
  auto sourceList = o2::event_visualisation::DirectoryLoader::load(src, "_", vExt1);
  std::vector<std::string> vExt2 = {ext2};
  auto destinationList = o2::event_visualisation::DirectoryLoader::load(dst, "_", vExt2);

  // first delete destination files which has not corresponding source files
  for (auto& e : destinationList) {
    auto match = e.substr(0, e.size() - ext2.size()) + ext1;
    if (sourceList.end() == std::find(sourceList.begin(), sourceList.end(), match)) {
      auto path = std::filesystem::path(dst + "" + e);
      std::filesystem::remove(path);
    }
  }

  // second translate source files which has not corresponding destination files
  for (auto& e : sourceList) {
    auto match = e.substr(0, e.size() - ext1.size()) + ext2;
    if (destinationList.end() == std::find(destinationList.begin(), destinationList.end(), match)) {
      // LOGF(info, "translate %s ->%s", src+e, dst+match);
      singleFileConversion(src + e, dst + match);
    }
  }

  return 0;
}

void my_handler(int s)
{
  printf("Caught signal %d\n", s);
  exit(1);
}

int main(int argc, char** argv)
{
  struct sigaction sigIntHandler {
  };
  sigIntHandler.sa_handler = my_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;

  sigaction(SIGINT, &sigIntHandler, nullptr);
  LOGF(info, "Welcome in O2 event conversion tool");

  if (argc == 3) {
    singleFileConversion(argv[1], argv[2]); // std::quick_exit(...
    return 0;
  }
  if (argc == 4 and std::string(argv[1]) == std::string("-l")) {
    singleFileConversion(argv[2], argv[3], 3); // std::quick_exit(...
    return 0;
  }
  if (argc == 4 and std::string(argv[1]) == std::string("-f")) {
    folderConversion(argv[2], argv[3]); // std::quick_exit(...
    return 0;
  }
  if (argc == 4 and std::string(argv[1]) == std::string("-c")) {
    while (true) {
      std::this_thread::sleep_for(2000ms);
      folderConversion(argv[2], argv[3]);
    }
    return 0;
  }
  LOGF(error, "two filename required, second should point to not existent file");
  return -1; //std::quick_exit(-1);
}
