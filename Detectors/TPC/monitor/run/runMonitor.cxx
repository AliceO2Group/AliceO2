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
/// \file   runSim.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

#include <boost/program_options.hpp>
#include <iostream>

#include "TRint.h"

#include "TPCMonitor/SimpleEventDisplayGUI.h"

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  // Arguments parsing
  std::string file;
  std::string pedestalFile;
  int lastTimeBin{512};
  int firstTimeBin{0};
  int verbosity{0};
  int debugLevel{0};
  int sector{0};
  bool overview{true};

  po::variables_map vm;
  po::options_description desc("Allowed options");
  desc.add_options()                                                                                 //
    ("fileInfo,i", po::value<std::string>(&file)->required(), "input file(s)")                       //
    ("pedestalFile,p", po::value<std::string>(&pedestalFile), "pedestal file")                       //
    ("firstTimeBin,f", po::value<int>(&firstTimeBin)->default_value(0), "first time bin to process") //
    ("lastTimeBin,l", po::value<int>(&lastTimeBin)->default_value(512), "last time bin to process")  //
    ("verbosity,v", po::value<int>(&verbosity)->default_value(0), "verbosity level")                 //
    ("debugLevel,d", po::value<int>(&debugLevel)->default_value(0), "debug level")                   //
    ("sector,s", po::value<int>(&sector)->default_value(0), "sector to be shown on startup")         //
    ("overview,o", po::value<bool>(&overview)->default_value(true), "show sides overview")           //
    ("help,h", "Produce help message.")                                                              //
    ;                                                                                                //

  po::store(parse_command_line(argc, argv, desc), vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << '\n';
    return EXIT_SUCCESS;
  }

  po::notify(vm);

  std::cout << "####" << '\n';
  std::cout << "#### Starting TPC simple online monitor" << '\n';
  std::cout << "#### filename: " << file << '\n';
  std::cout << "####" << '\n';
  std::cout << '\n'
            << '\n';

  TRint rootApp("TPC Event Monitor", nullptr, nullptr);

  o2::tpc::SimpleEventDisplayGUI g;
  g.runSimpleEventDisplay(file + ":" + std::to_string(lastTimeBin), pedestalFile, firstTimeBin, lastTimeBin, lastTimeBin, verbosity, debugLevel, sector, overview);

  rootApp.Run(true);

  return EXIT_SUCCESS;
}
