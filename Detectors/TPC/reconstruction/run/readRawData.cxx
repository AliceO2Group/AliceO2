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
/// @file   readRawData.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <memory>
#include <bitset>
#include <vector>

#include "TPCReconstruction/RawReader.h"
#include "TPCReconstruction/RawReaderEventSync.h"
#include "TPCBase/PadPos.h"
#include "FairLogger.h"

namespace bpo = boost::program_options;
using namespace o2::TPC;

int main(int argc, char *argv[])
{

  // Arguments parsing
  std::vector<std::string> infile(1,"NOFILE");
  std::vector<unsigned> region(1,0);
  std::vector<unsigned> link(1,0);
  int readEvents = -1;
  bool useRawInMode3 = true;
  std::string verbLevel = "LOW";
  std::string logLevel = "ERROR";

  bpo::variables_map vm;
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("infile,i",    bpo::value<std::vector<std::string>>(&infile),   "Input data files")
    ("vl,",         bpo::value<std::string>(&verbLevel),"Fairlogger verbosity level (LOW, MED, HIGH)")
    ("ll,",         bpo::value<std::string>(&logLevel), "Fairlogger screen log level (FATAL, ERROR, WARNING, INFO, DEBUG, DEBUG1, DEBUG2, DEBUG3, DEBUG4)")
    ("region,r",    bpo::value<std::vector<unsigned>>(&region),      "Region for mapping")
    ("link,l",      bpo::value<std::vector<unsigned>>(&link),        "Link for mapping")
    ("rawInMode3",  bpo::value<bool>(&useRawInMode3),   "Use Raw data in mode 3 instead of decoded one")
    (",n",          bpo::value<int>(&readEvents),       "Events to read");

  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }

  if (infile[0] == "NOFILE") return EXIT_SUCCESS;

  // Initialize logger
  FairLogger *logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel(verbLevel.c_str());
  logger->SetLogScreenLevel(logLevel.c_str());
  logger->SetColoredLog(false);

  std::vector<RawReader> readers;
  std::shared_ptr<RawReaderEventSync> eventSync = std::make_shared<RawReaderEventSync>();

  for (int i = 0; i < infile.size(); ++i) {
    readers.emplace_back();
    readers[i].addEventSynchronizer(eventSync);
    if (region.size() != infile.size() && link.size() == infile.size())
      readers[i].addInputFile(region[0],link[i],infile[i]);
    else if (region.size() == infile.size() && link.size() != infile.size())
      readers[i].addInputFile(region[i],link[0],infile[i]);
    else
      readers[i].addInputFile(region[i],link[i],infile[i]);
    readers[i].setUseRawInMode3(useRawInMode3);
    readers[i].setCheckAdcClock(false);
  }

  PadPos padPos;
  int j = 0;
  while((j < readers[0].getNumberOfEvents()) & ((readEvents>=0)? j<=readEvents : true)) {
    for (auto &rr : readers) {
      rr.loadEvent(j);
    }
    ++j;
  }

  return EXIT_SUCCESS;
}
