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
/// \file    Options.cxx
/// \author  Julian Myrcha

#include "EventVisualisationView/Options.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include <fairlogger/Logger.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

namespace o2
{
namespace event_visualisation
{

Options Options::instance;

namespace bpo = boost::program_options;

std::string Options::printOptions()
{
  static const char* str[2] = {"false", "true"};
  std::stringstream ss;
  ss << "fileName    : " << this->fileName() << std::endl;
  ss << "data folder : " << this->dataFolder() << std::endl;
  ss << "saved data folder : " << this->savedDataFolder() << std::endl;
  ss << "randomTracks: " << str[this->randomTracks()] << std::endl;
  ss << "json        : " << str[this->json()] << std::endl;
  ss << "online      : " << str[this->online()] << std::endl;
  return ss.str();
}

bool Options::processCommandLine(int argc, char* argv[])
{
  bool save = false;

  bpo::options_description eveOptions("o2Eve options");

  eveOptions.add_options()(
    "help,h", "produce help message")(
    "datafolder,d", bpo::value<decltype(this->mDataFolder)>()->default_value("./json"), "name of the data folder")(
    "imagefolder,i", bpo::value<decltype(this->mImageFolder)>()->default_value(""), "name of the image folder")(
    "filename,f", bpo::value<decltype(this->mFileName)>()->default_value("data.root"), "name of the data file")(
    "json,j", bpo::value<decltype(this->mJSON)>()->zero_tokens()->default_value(false), "use json files as a source")(
    "memorylimit,m", bpo::value<decltype(this->mMemoryLimit)>()->default_value(-1), "memory usage limit (MB) - app will terminate if it is exceeded (pass -1 for no limit)")(
    "online,o", bpo::value<decltype(this->mOnline)>()->zero_tokens()->default_value(false), "use online json files as a source")(
    "optionsfilename,p", bpo::value<std::string>()->default_value("o2eve.json"), "name of the options file")(
    "randomtracks,r", bpo::value<decltype(this->mRandomTracks)>()->zero_tokens()->default_value(false), "use random tracks")(
    "saveddatafolder,s", bpo::value<decltype(this->mSavedDataFolder)>()->default_value(""), "name of the saved data folder")(
    "hidedplgui", bpo::value<decltype(this->mHideDplGUI)>()->zero_tokens()->default_value(false), "hide DPL GUI when processing AODs")(
    "aodconverter,a", bpo::value<decltype(this->mAODConverterPath)>()->default_value("o2-eve-aodconverter"), "AOD converter path");

  using namespace bpo::command_line_style;
  auto style = (allow_short | short_allow_adjacent | short_allow_next | allow_long | long_allow_adjacent | long_allow_next | allow_sticky | allow_dash_for_short);
  bpo::variables_map varmap;
  try {
    bpo::store(
      bpo::command_line_parser(argc, argv)
        .options(eveOptions)
        .style(style)
        .run(),
      varmap);
  } catch (std::exception const& e) {
    LOGP(error, "error parsing options of {}: {}", argv[0], e.what());
    exit(1);
  }

  if (varmap.count("help")) {
    LOG(info) << eveOptions << std::endl
              << "  default values are always taken from o2eve.json in current folder if present" << std::endl;
    return false;
  }

  this->mDataFolder = varmap["datafolder"].as<decltype(this->mDataFolder)>();
  this->mImageFolder = varmap["imagefolder"].as<decltype(this->mImageFolder)>();
  this->mFileName = varmap["filename"].as<decltype(this->mFileName)>();
  this->mJSON = varmap["json"].as<decltype(this->mJSON)>();
  this->mMemoryLimit = varmap["memorylimit"].as<decltype(this->mMemoryLimit)>();
  this->mOnline = varmap["online"].as<decltype(this->mOnline)>();
  auto optionsFileName = varmap["optionsfilename"].as<std::string>();
  this->mRandomTracks = varmap["randomtracks"].as<decltype(this->mRandomTracks)>();
  this->mSavedDataFolder = varmap["saveddatafolder"].as<decltype(this->mSavedDataFolder)>();
  this->mHideDplGUI = varmap["hidedplgui"].as<decltype(this->mHideDplGUI)>();
  this->mAODConverterPath = varmap["aodconverter"].as<decltype(this->mAODConverterPath)>();

  if (save) {
    this->saveToJSON("o2eve.json");
    return false;
  }

  return true;
}

bool Options::saveToJSON(std::string filename)
{
  rapidjson::Value dataFolder;
  rapidjson::Value savedDataFolder;
  rapidjson::Value fileName;
  rapidjson::Value json(rapidjson::kNumberType);
  rapidjson::Value online(rapidjson::kNumberType);
  rapidjson::Value randomTracks(rapidjson::kNumberType);

  dataFolder.SetString(rapidjson::StringRef(this->dataFolder().c_str()));
  savedDataFolder.SetString(rapidjson::StringRef(this->savedDataFolder().c_str()));
  fileName.SetString(rapidjson::StringRef(this->fileName().c_str()));
  json.SetBool(this->json());
  online.SetBool(this->online());
  randomTracks.SetBool(this->randomTracks());

  rapidjson::Document tree(rapidjson::kObjectType);
  rapidjson::Document::AllocatorType& allocator = tree.GetAllocator();
  tree.AddMember("dataFolder", dataFolder, allocator);
  tree.AddMember("savedDataFolder", savedDataFolder, allocator);
  tree.AddMember("fileName", fileName, allocator);
  tree.AddMember("json", json, allocator);
  tree.AddMember("online", online, allocator);
  tree.AddMember("randomTracks", randomTracks, allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  tree.Accept(writer);

  std::ofstream out(filename);
  out << buffer.GetString();
  out.close();
  return true;
}

bool Options::readFromJSON(std::string /*filename*/)
{
  return false;
}

} // namespace event_visualisation
} // namespace o2
