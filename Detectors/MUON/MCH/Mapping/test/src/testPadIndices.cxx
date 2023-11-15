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

#include "InputDocument.h"
#include <string>
#include <iostream>
#include <fmt/format.h>
#include "MCHMappingInterface/Segmentation.h"
#include <boost/program_options.hpp>
#include <vector>
#include <fstream>

namespace po = boost::program_options;

std::string asString(rapidjson::Value& tp)
{
  //got     =FEC  229CH 25X      19Y      20SX     2.5SY     0.5 (B)
  return fmt::format("FEC {} CH {} padindex {} bending {} X {} Y {}",
                     tp["dsid"].GetInt(),
                     tp["dsch"].GetInt(),
                     tp["padindex"].GetInt(), tp["bending"].GetString(), tp["x"].GetDouble(),
                     tp["y"].GetDouble());
}

int CheckPadIndexIsAsExpected(std::string filepath)
{
  InputWrapper data(filepath.c_str());

  rapidjson::Value& padIndices = data.document()["channels"];

  int nbad{0};
  int ntested{0};

  for (auto& tp : padIndices.GetArray()) {
    ntested++;
    int detElemId = tp["de"].GetInt();
    bool isBendingPlane = (tp["bending"].GetString() == std::string("true"));
    int refPadIndex = tp["padindex"].GetInt();
    const auto& seg = o2::mch::mapping::segmentation(detElemId);
    int bpad;
    int nbpad;
    seg.findPadPairByPosition(tp["x"].GetDouble(), tp["y"].GetDouble(), bpad, nbpad);
    int padIndex = (isBendingPlane ? bpad : nbpad);
    if (padIndex != refPadIndex) {
      nbad++;
      std::cout << fmt::format(">>> {} : Expected index = {} but got {}\n", filepath.c_str(), refPadIndex, padIndex);
      std::cout << "expected=" << seg.padAsString(refPadIndex) << "\n";
      std::cout << "got     =" << seg.padAsString(padIndex) << "\n";
      std::cout << "b       =" << seg.padAsString(bpad) << "\n";
      std::cout << "nb      =" << seg.padAsString(nbpad) << "\n";
    }
  }

  if (nbad) {
    std::cout << fmt::format("\n{} : {} pad indices error(s) over {} tested pads\n\n", filepath.c_str(), nbad, ntested);
  } else {
    std::cout << fmt::format("\n{} : the indices of all tested {} pads are OK\n\n", filepath.c_str(), ntested);
  }
  return nbad;
}

int main(int argc, char** argv)
{
  std::string filePattern;
  std::vector<int> deIds;
  po::variables_map vm;
  po::options_description generic("Generic options");

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("filepattern,p", po::value<std::string>(&filePattern)->required(), "input file pattern")
      ("de,d",po::value<std::vector<int>>(&deIds)->multitoken(),"detection element")
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

  int nbad{0};

  for (auto de : deIds) {
    std::string filepath(fmt::format(fmt::runtime(filePattern), de));
    std::ifstream in(filepath);
    if (!in) {
      std::cout << "Cannot open " << filepath << "\n";
      return -1;
    }
    nbad += CheckPadIndexIsAsExpected(filepath);
  }
  return nbad != 0;
}
