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
/// @file   testRawReader.cxx
/// @author Sebastian Klewin
///

#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <bitset>
#include <vector>
#include <memory>

#include "TPCReconstruction/RawReader.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadPos.h"
#include "FairLogger.h"

namespace bpo = boost::program_options; 

int main(int argc, char *argv[])
{
  // Initialize logger
  FairLogger *logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("DEBUG");


  // Arguments parsing
  std::vector<std::string> infiles;
  int readFrames = -1;

  bpo::variables_map vm; 
  bpo::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("infile,i",    bpo::value<std::vector<std::string>>(&infiles),   "Input data files")
    (",n",          bpo::value<int>(&readFrames),       "Frames to read");

  bpo::store(parse_command_line(argc, argv, desc), vm);
  bpo::notify(vm);

  // help
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return EXIT_SUCCESS;
  }


  o2::TPC::RawReader rr;
  rr.setUseRawInMode3(false);
  rr.addInputFile(&infiles);

//  or 
//
//  for (auto &i : infiles)
//    rr.addInputFile(i);


//  const o2::TPC::Mapper& mapper = o2::TPC::Mapper::instance();
//  for (int i=0; i<100; ++i) {
//    rr.loadNextEvent();
//  }
  uint64_t ts = 0;
  std::cout << "First event: " << rr.getFirstEvent() << " Last event: " << rr.getLastEvent() << " number of events available: " << rr.getNumberOfEvents() << std::endl;
  for (int i=rr.getFirstEvent(); i<=rr.getLastEvent(); ++i) {
    std::cout << "i: " << i << " loaded Event: " << rr.loadEvent(i) << std::endl;
//    std::cout << rr.getTimeStamp(1) << " " << rr.getTimeStamp(1) - ts << std::endl;
//    ts = rr.getTimeStamp(1);
    o2::TPC::PadPos padPos;
    while (std::shared_ptr<std::vector<uint16_t>> data = rr.getNextData(padPos)) {
      std::cout << "Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << " " << data->size() << std::endl;
      //for (std::vector<uint16_t>::iterator it = data->begin(); it != data->end(); ++it) {
      //  std::cout << *it << std::endl;
      //}
    }
  }

//  std::cout << "part 1 done" << std::endl;
//  for (int sampa = 0; sampa < 3; ++sampa) {
//    for (int channel = 0; channel < 32; ++channel) {
//      o2::TPC::PadPos padPos = mapper.padPosRegion(0,0,sampa,channel);
//      std::shared_ptr<std::vector<uint16_t>> data = rr.getData(padPos);
//
//      std::cout << "S: " << sampa << " C: " << channel << " Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << " " << data->size() << std::endl;
//      for (std::vector<uint16_t>::iterator it = data->begin(); it != data->end(); ++it) {
//        std::cout << *it << std::endl;
//      }
//    }
//  }


  return EXIT_SUCCESS;
}
