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

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/ostreamwrapper.h"
#include <iostream>
#include <fmt/format.h>
#include "MCHMappingInterface/Segmentation.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char** argv)
{
  std::string filePattern;
  int deId;
  po::variables_map vm;
  po::options_description generic("Generic options");
  bool sparse{false};

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("de,d",po::value<int>(&deId)->required(),"detection element")
      ("sparse,s",po::bool_switch(&sparse),"do not generate all indices but only a sample")
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

  rapidjson::OStreamWrapper osw(std::cout);
  rapidjson::Writer<rapidjson::OStreamWrapper> w(osw);

  o2::mch::mapping::Segmentation seg{deId};

  w.StartObject();
  w.Key("channels");
  w.StartArray();

  int n{0};

  seg.forEachPad([&](const int& padindex) {
    n++;
    if (sparse && (n % 16)) {
      return;
    }
    w.StartObject();
    w.Key("de");
    w.Int(deId);
    w.Key("bending");
    w.String(seg.isBendingPad(padindex) ? "true" : "false");
    w.Key("x");
    w.Double(seg.padPositionX(padindex));
    w.Key("y");
    w.Double(seg.padPositionY(padindex));
    w.Key("padindex");
    w.Int(padindex);
    w.Key("dsid");
    w.Int(seg.padDualSampaId(padindex));
    w.Key("dsch");
    w.Int(seg.padDualSampaChannel(padindex));
    w.EndObject();
  });

  w.EndArray();
  w.EndObject();
  return 0;
}
