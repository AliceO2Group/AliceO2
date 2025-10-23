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
#include <EventVisualisationDataConverter/Location.h>
#include <EventVisualisationDataConverter/VisualisationEventJSONSerializer.h>
#include <EventVisualisationDataConverter/VisualisationEventROOTSerializer.h>
#include <EventVisualisationDataConverter/VisualisationEventOpenGLSerializer.h>
#include <EventVisualisationBase/DirectoryLoader.h>
#include <TEveManager.h>
#include <filesystem>
#include <fairlogger/Logger.h>
#include <csignal>
#include <thread>
#include <chrono>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std::chrono_literals;

// source file name, destination (not existing) file name, if limit > 0 then limit EACH type of data
int singleFileConversion(const std::string& src, o2::event_visualisation::Location& dst, const int limit = -1)
{
  LOGF(info, "Translate: %s -> %s", src, dst.fileName());
  o2::event_visualisation::VisualisationEvent vEvent;

  auto srcSerializer = o2::event_visualisation::VisualisationEventSerializer::getInstance(
    std::filesystem::path(src).extension());
  auto dstExtension = std::filesystem::path(
                        src)
                        .extension(); // if there is no destination, there will be no extension change
  if (!dst.fileName().empty()) {
    dstExtension = std::filesystem::path(dst.fileName()).extension();
  }
  auto dstSerializer = o2::event_visualisation::VisualisationEventSerializer::getInstance(
    dstExtension);

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
int folderConversion(const std::string& srcFolder, const o2::event_visualisation::Location& dstFolderLocation)
{
  const std::string dstFolder = dstFolderLocation.fileName();
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
      o2::event_visualisation::Location location({.fileName = dst + match,
                                                  .port = dstFolderLocation.port(),
                                                  .host = dstFolderLocation.hostName()});
      singleFileConversion(src + e, location);
      ;
      singleFileConversion(src + e, location);
      ;
    }
  }

  return 0;
}

void my_handler(int s)
{
  printf("Caught signal %d\n", s);
  exit(1);
}

namespace po = boost::program_options;

using namespace std;

int main(int argc, char** argv)
{
  struct sigaction sigIntHandler {
  };
  sigIntHandler.sa_handler = my_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;

  sigaction(SIGINT, &sigIntHandler, nullptr);
  LOGF(info, "Welcome in O2 event conversion tool");

  try {
    int port;
    string host;
    int limit;
    bool folderMode;
    bool continuousMode;
    vector<string> sources;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")("port", po::value(&port)->default_value(-1), "port number")("host", po::value(&host)->default_value("localhost"), "host name")("sources", po::value(&sources), "sources")("limit,l", po::value(&limit)->default_value(-1), "limit number of elements")("folder,f", po::bool_switch(&folderMode)->default_value(false), "convert folders")("continuous,c", po::bool_switch(&continuousMode)->default_value(false), "continuous folder mode");

    po::positional_options_description p;
    p.add("sources", 2);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      return 0;
    }

    if (vm.count("sources")) {
      if (vm["sources"].as<vector<string>>().size() != 2) {
        cout << "two positional parameters expected" << "\n";
        return 0;
      }
    }
    o2::event_visualisation::LocationParams locationParams;
    locationParams.fileName = sources[1];
    locationParams.port = port;
    locationParams.host = host;

    o2::event_visualisation::Location location(locationParams);

    if (folderMode) {
      folderConversion(sources[0], location);
    } else if (continuousMode) {
      while (true) {
        std::this_thread::sleep_for(2000ms);
        folderConversion(sources[0], location);
      }
    } else {
      singleFileConversion(sources[0], location, limit);
      return 0;
    }
  }

  catch (exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    cerr << "Exception of unknown type!\n";
  }
  return -1; // std::quick_exit(-1);
}
