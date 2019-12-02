// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   mid-rawdump.cxx
/// \brief  Raw dumper for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 December 2019

#include <iostream>
#include "boost/program_options.hpp"
#include "MIDRaw/CRUBareDecoder.h"
#include "MIDRaw/RawFileReader.h"

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");
  std::string outFilename = "mid_raw.txt";
  unsigned long int nHBs = 0;
  bool onlyClosedHBs = false;

  // clang-format off
  generic.add_options()
          ("help", "produce help message")
          ("output", po::value<std::string>(&outFilename),"Output text file")
          ("nHBs", po::value<unsigned long int>(&nHBs),"Number of HBs read")
          ("only-closed-HBs", po::value<bool>(&onlyClosedHBs)->implicit_value(true),"Return only closed HBs");


  po::options_description hidden("hidden options");
  hidden.add_options()
          ("input", po::value<std::vector<std::string>>(),"Input filename");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic).add(hidden);

  po::positional_options_description pos;
  pos.add("input", -1);

  po::store(po::command_line_parser(argc, argv).options(cmdline).positional(pos).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << "Usage: " << argv[0] << " <input_raw_filename> [input_raw_filename_1 ...]\n";
    std::cout << generic << std::endl;
    return 2;
  }
  if (vm.count("input") == 0) {
    std::cout << "no input file specified" << std::endl;
    return 1;
  }

  std::vector<std::string> inputfiles{vm["input"].as<std::vector<std::string>>()};

  o2::mid::CRUBareDecoder decoder;

  std::ofstream outFile(outFilename);

  unsigned long int iHB = 0;

  for (auto& filename : inputfiles) {
    o2::mid::RawFileReader<uint8_t> rawFileReader;
    if (!rawFileReader.init(filename.c_str())) {
      return 2;
    }
    while (rawFileReader.getState() == 0) {
      rawFileReader.readHB(onlyClosedHBs);
      decoder.process(rawFileReader.getData());
      rawFileReader.clear();
      for (auto& rof : decoder.getROFRecords()) {
        outFile << "Orbit: " << rof.interactionRecord.orbit << " bc: " << rof.interactionRecord.bc << std::endl;
        for (auto colIt = decoder.getData().begin() + rof.firstEntry; colIt != decoder.getData().begin() + rof.firstEntry + rof.nEntries; ++colIt) {
          outFile << *colIt << std::endl;
        }
      }
      ++iHB;
      if (nHBs > 0 && iHB >= nHBs) {
        break;
      }
    }
  }
  outFile.close();

  return 0;
}
