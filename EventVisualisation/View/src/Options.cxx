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
/// \file    Options.cxx
/// \author  Julian Myrcha

#include "EventVisualisationView/Options.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include "FairLogger.h"
#include <unistd.h>
#include <iostream>
#include <string>
#include <sstream>

namespace o2
{
namespace event_visualisation
{

Options Options::instance;

std::string Options::printOptions()
{
  static const char* str[2] = {"false", "true"};
  std::stringstream ss;
  ss << "fileName    : " << this->fileName() << std::endl;
  ss << "randomTracks: " << str[this->randomTracks()] << std::endl;
  ss << "itc         : " << str[this->its()] << std::endl;
  ss << "json        : " << str[this->json()] << std::endl;
  ss << "online      : " << str[this->online()] << std::endl;
  ss << "vsd         : " << str[this->vsd()] << std::endl;
  ss << "tpc         : " << str[this->tpc()] << std::endl;
  return ss.str();
}

std::string Options::usage()
{
  std::stringstream ss;
  ss << "usage:" << std::endl;
  ss << "\t"
     << "o2eve <options>" << std::endl;
  ss << "\t\t"
     << "where <options> are any from the following:" << std::endl;
  ss << "\t\t"
     << "-h             this help message" << std::endl;
  ss << "\t\t"
     << "-d name        name of the data folder" << std::endl;
  ss << "\t\t"
     << "-f name        name of the data file" << std::endl;
  ss << "\t\t"
     << "-i             use itc reading from files as a source" << std::endl;
  ss << "\t\t"
     << "-j             use json files as a source" << std::endl;
  ss << "\t\t"
     << "-o             use online json files as a source" << std::endl;
  ss << "\t\t"
     << "-p name        name of the options file" << std::endl;
  ss << "\t\t"
     << "-r             use random tracks" << std::endl;
  ss << "\t\t"
     << "-s             save options to o2eve.json in current folder" << std::endl;
  ss << "\t\t"
     << "-t             use tpc reading from files as a source" << std::endl;
  ss << "\t\t"
     << "-v             use vsd files as a source" << std::endl;
  ss << "\tdefault values are always taken from o2eve.json in current folder if present" << std::endl;
  return ss.str();
}

bool Options::processCommandLine(int argc, char* argv[])
{
  int opt;
  bool save = false;
  std::string optionsFileName = "o2eve.json"; // name with options to use

  // put ':' in the starting of the
  // string so that program can
  //distinguish between '?' and ':'
  while ((opt = getopt(argc, argv, ":d:f:hijop:rsvt")) != -1) {
    switch (opt) {
      case 'd':
        this->mDataFolder = optarg;
        break;
      case 'f':
        this->mFileName = optarg;
        break;
      case 'i':
        this->mIts = true;
        break;
      case 'h':
        std::cout << usage() << std::endl;
        return false;
      case 'j':
        this->mJSON = true;
        break;
      case 'o':
        this->mOnline = true;
        break;
      case 'p':
        optionsFileName = optarg;
        break;
      case 'r':
        this->mRandomTracks = true;
        break;
      case 's':
        save = true;
        break;
      case 't':
        this->mTpc = true;
        break;
      case 'v':
        this->mVsd = true;
        break;
      case ':':
        LOG(ERROR) << "option needs a value: " << char(optopt);
        LOG(INFO) << usage();
        return false;
      case '?':
        LOG(ERROR) << "unknown option: " << char(optopt);
        LOG(INFO) << usage();
        return false;
    }
  }

  // optind is for the extra arguments
  // which are not parsed
  for (; optind < argc; optind++) {
    LOG(ERROR) << "extra arguments: " << argv[optind];
    LOG(INFO) << usage();
    return false;
  }

  if (save) {
    this->saveToJSON("o2eve.json");
    return false;
  }

  return true;
}

bool Options::saveToJSON(std::string filename)
{
  rapidjson::Value dataFolder;
  rapidjson::Value fileName;
  rapidjson::Value its(rapidjson::kNumberType);
  rapidjson::Value json(rapidjson::kNumberType);
  rapidjson::Value online(rapidjson::kNumberType);
  rapidjson::Value randomTracks(rapidjson::kNumberType);
  rapidjson::Value tpc(rapidjson::kNumberType);
  rapidjson::Value vsd(rapidjson::kNumberType);

  dataFolder.SetString(rapidjson::StringRef(this->dataFolder().c_str()));
  fileName.SetString(rapidjson::StringRef(this->fileName().c_str()));
  its.SetBool(this->its());
  json.SetBool(this->json());
  online.SetBool(this->online());
  randomTracks.SetBool(this->randomTracks());
  tpc.SetBool(this->tpc());
  vsd.SetBool(this->vsd());

  rapidjson::Document tree(rapidjson::kObjectType);
  rapidjson::Document::AllocatorType& allocator = tree.GetAllocator();
  tree.AddMember("dataFolder", dataFolder, allocator);
  tree.AddMember("fileName", fileName, allocator);
  tree.AddMember("its", its, allocator);
  tree.AddMember("json", json, allocator);
  tree.AddMember("online", online, allocator);
  tree.AddMember("randomTracks", randomTracks, allocator);
  tree.AddMember("tpc", tpc, allocator);
  tree.AddMember("vsd", vsd, allocator);

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
