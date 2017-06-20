// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   readRawData.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <bitset>
#include <vector>

#include "TPCReconstruction/HalfSAMPAData.h"
#include "TPCReconstruction/RawReader.h"
#include "FairLogger.h"

namespace bpo = boost::program_options; 

int main(int argc, char *argv[])
{

  // Arguments parsing
  std::string infile("NOFILE");
  unsigned region = 0;
  unsigned link = 0;
  int readEvents = -1;
  bool useRawInMode3 = false;
  std::string verbLevel = "LOW";
  std::string logLevel = "ERROR";

  bpo::variables_map vm; 
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("infile,i",    bpo::value<std::string>(&infile),   "Input data files")
    ("vl,",         bpo::value<std::string>(&verbLevel),"Fairlogger verbosity level (LOW, MED, HIGH)")
    ("ll,",         bpo::value<std::string>(&logLevel), "Fairlogger screen log level (FATAL, ERROR, WARNING, INFO, DEBUG, DEBUG1, DEBUG2, DEBUG3, DEBUG4)")
    ("region,r",    bpo::value<unsigned>(&region),      "Region for mapping")
    ("link,l",      bpo::value<unsigned>(&link),        "Link for mapping")
    ("rawInMode3",  bpo::value<bool>(&useRawInMode3),   "Use Raw data in mode 3 instead of decoded one")
    (",n",          bpo::value<int>(&readEvents),       "Events to read");

  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  if (infile == "NOFILE") return EXIT_SUCCESS;

  // Initialize logger
  FairLogger *logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel(verbLevel.c_str());
  logger->SetLogScreenLevel(logLevel.c_str());


  o2::TPC::RawReader rr;
  rr.addInputFile(region,link,infile);
  rr.setUseRawInMode3(useRawInMode3);
  rr.setCheckAdcClock(false);
  rr.setPrintRawData(true);

  int i=0;
  while((rr.loadNextEventNoWrap() >= 0) & ((readEvents>=0)? i<readEvents : true)) {
    ++i;
  }

  return EXIT_SUCCESS;
}
