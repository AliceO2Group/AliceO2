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

/// \file   MID/GlobalMapping/exe/global-mapper.cxx
/// \brief  Generate MID mapping files
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 April 2023

#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <vector>
#include <fmt/format.h>
#include "boost/program_options.hpp"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include "MIDBase/DetectorParameters.h"
#include "MIDGlobalMapping/ExtendedMappingInfo.h"
#include "MIDGlobalMapping/GlobalMapper.h"

namespace po = boost::program_options;

void stripsInfo2json(const std::vector<o2::mid::ExtendedMappingInfo>& infos, const char* outDir)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  writer.StartArray();
  for (size_t idx = 0; idx < infos.size(); ++idx) {
    writer.StartObject();
    writer.Key("id");
    writer.Int(infos[idx].id);
    writer.Key("idx");
    writer.Int(idx);
    writer.Key("rpc");
    writer.String(infos[idx].rpc.c_str());
    writer.Key("deId");
    writer.Int(infos[idx].deId);
    writer.Key("columnId");
    writer.Int(infos[idx].columnId);
    writer.Key("lineId");
    writer.Int(infos[idx].lineId);
    writer.Key("stripId");
    writer.Int(infos[idx].stripId);
    writer.Key("cathode");
    writer.Int(infos[idx].cathode);
    writer.Key("locId");
    writer.Int(infos[idx].locId);
    writer.Key("locIdDcs");
    writer.String(infos[idx].locIdDcs.c_str());
    writer.EndObject();
  }
  writer.EndArray();

  std::ofstream outInfo(fmt::format("{}/stripInfo.json", outDir));
  outInfo << buffer.GetString() << std::endl;
}

void strips2json(const std::vector<o2::mid::ExtendedMappingInfo>& infos, const char* outDir)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

  auto strip2json = [&](const o2::mid::ExtendedMappingInfo& info) {
    writer.StartObject();
    writer.Key("id");
    writer.Int(info.id);
    writer.Key("vertices");
    writer.StartArray();
    // (x1,y1)
    writer.StartObject();
    writer.Key("x");
    writer.Int(info.xpos);
    writer.Key("y");
    writer.Int(info.ypos);
    writer.EndObject();
    // (x2,y1)
    writer.StartObject();
    writer.Key("x");
    writer.Int(info.xpos + info.xwidth);
    writer.Key("y");
    writer.Int(info.ypos);
    writer.EndObject();
    // (x2,y2)
    writer.StartObject();
    writer.Key("x");
    writer.Int(info.xpos + info.xwidth);
    writer.Key("y");
    writer.Int(info.ypos + info.ywidth);
    writer.EndObject();
    // (x1,y2)
    writer.StartObject();
    writer.Key("x");
    writer.Int(info.xpos);
    writer.Key("y");
    writer.Int(info.ypos + info.ywidth);
    writer.EndObject();
    // (x1,y1)
    writer.StartObject();
    writer.Key("x");
    writer.Int(info.xpos);
    writer.Key("y");
    writer.Int(info.ypos);
    writer.EndObject();
    writer.EndArray();
    writer.EndObject();
  };

  std::array<std::string, 2> planes{"bending", "non-bending"};

  for (int ide = 0; ide < o2::mid::detparams::NDetectionElements; ++ide) {
    for (int icath = 0; icath < 2; ++icath) {
      std::string fname = fmt::format("{}/strips.{}.{}.json", outDir, ide, planes[icath]);
      writer.StartArray();
      for (auto& info : infos) {
        if (info.deId == ide && info.cathode == icath) {
          strip2json(info);
        }
      }
      writer.EndArray();
      std::ofstream outFile(fname);
      outFile << buffer.GetString() << std::endl;
      buffer.Clear();
    }
  }
}

void de2json(const std::map<int, std::vector<std::pair<int, int>>>& deMap, const char* outDir)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  for (auto& item : deMap) {
    buffer.Clear();
    writer.StartObject();
    writer.Key("deId");
    writer.Int(item.first);
    writer.Key("vertices");
    writer.StartArray();
    for (auto& vtx : item.second) {
      writer.StartObject();
      writer.Key("x");
      writer.Int(vtx.first);
      writer.Key("y");
      writer.Int(vtx.second);
      writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();

    std::string fname = fmt::format("{}/de.{}.json", outDir, item.first);
    std::ofstream outFile(fname);
    outFile << buffer.GetString() << std::endl;
  }
}

void writeJson(const std::vector<o2::mid::ExtendedMappingInfo>& infos, const std::map<int, std::vector<std::pair<int, int>>>& deMap, const char* outDir)
{

  if (!std::filesystem::exists(outDir)) {
    std::filesystem::create_directory(outDir);
  }
  stripsInfo2json(infos, outDir);
  strips2json(infos, outDir);
  de2json(deMap, outDir);
}

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");

  std::string outDir;

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("outdir,o", po::value<std::string>(&outDir)->default_value("midmapping"), "Output directory")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << generic << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  o2::mid::GlobalMapper gm;
  gm.setScaleFactor(2);
  auto infos = gm.buildStripsInfo();
  auto deMap = gm.buildDEGeom();

  writeJson(infos, deMap, outDir.c_str());
  return 0;
}