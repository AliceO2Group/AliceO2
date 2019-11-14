// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file rawReaderFile.cxx 
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
/// \author Torsten Alt (Torsten.Alt@cern.ch)
/// \author Boris Polishchuk (Boris.Polishchuk@cern.ch, modification for PHOS)

#include <iostream>
#include <boost/program_options.hpp>

#include "EMCALReconstruction/RawReaderFile.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/Mapper.h"

namespace bpo = boost::program_options;
using namespace o2::emcal;

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " <cmds/options>\n"
                                       "  Tool will decode the GBTx data for SAMPA 0\n"
                                       "Commands / Options");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbose,v", bpo::value<uint32_t>()->default_value(0), "Select verbosity level [0 = no output]");
    add_option("version", "Print version information");
    add_option("input-file,i", bpo::value<std::string>()->required(), "Specifies input file.");
    add_option("debug,d", bpo::value<uint32_t>()->default_value(0), "Select debug output level [0 = no debug output]");

    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help") || argc == 1) {
      std::cout << opt_general << std::endl;
      exit(0);
    }

    if (vm.count("version")) {
      //std::cout << GitInfo();
      exit(0);
    }

    bpo::notify(vm);
  } catch (bpo::error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl
              << std::endl;
    std::cerr << opt_general << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << e.what() << ", application will now exit" << std::endl;
    exit(2);
  }

  int fNModules = 4; // number of PHOS modules
  int fNRCU     = 4; // number of CRUs per module

  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";

  if (aliceO2env)
    inputDir = aliceO2env;

  inputDir += "/share/Detectors/PHOS/files/";
  std::string prefix = inputDir + "Mod";

  std::vector<Mapper* > mappers;

  for(int sm = 1; sm <= fNModules; sm++) {

    auto path1 = prefix + std::to_string( sm );
    path1 += "RCU";

    for(int cru = 0; cru < fNRCU; cru++) {

      auto path2 = path1 + std::to_string( cru );
      path2 += ".data";

      std::cout << "Mapping file: "<< path2 << std::endl;

      Mapper* mp = new Mapper(path2);
      mappers.push_back(mp);
    }
  }


  // How to figure out DDL ID in O2??

  //Int_t AliRawReader::GetDDLID() const
  //{

  //  // Get the DDL ID (within one sub-detector)
  //  // The list of detector IDs
  //  // can be found in AliDAQ.h

  //Int_t equipmentId;

  //if (fEquipmentIdsIn && fEquipmentIdsIn)
  //equipmentId = GetMappedEquipmentId();
  //else
  // equipmentId = GetEquipmentId();

  //if (equipmentId >= 0) {
  //Int_t ddlIndex;
  //AliDAQ::DetectorIDFromDdlID(equipmentId,ddlIndex);
  //return ddlIndex;
  //}
  //else
  //return -1;
  //}

  //Int_t ddlNumber = GetDDLNumber();
  //fModule = ddlNumber / fNRCU;

  //Int_t rcuIndex = ddlNumber % fNRCU;
  //Short_t hwAddress = GetHWAddress();

  //if(rcuIndex > -1 && rcuIndex < 20 && hwAddress > -1) {
  //  fRow      = fMapping[rcuIndex]->GetPadRow(hwAddress);
  //  fColumn   = fMapping[rcuIndex]->GetPad(hwAddress);
  //  fCaloFlag = fMapping[rcuIndex]->GetSector(hwAddress);
  //}


  o2::emcal::RawReaderFile reader(vm["input-file"].as<std::string>());
  auto mapper = mappers.at(2); // CRU2

  for (int ipage = 0; ipage < reader.getNumberOfPages(); ipage++) {
    reader.nextPage();
    std::cout << reader.getRawHeader();
    o2::emcal::AltroDecoder decoder(reader);
    decoder.decode();
    std::cout << decoder.getRCUTrailer() << std::endl;
    for (auto& chan : decoder.getChannels()) {
      std::cout << "Hw address: " << chan.getHardwareAddress() << std::endl;
      for (auto& bunch : chan.getBunches()) {
        std::cout << "BunchLength: " << int(bunch.getBunchLength()) << std::endl;
        auto adcs = bunch.getADC();
        int time = bunch.getStartTime();
        for (int i = adcs.size() - 1; i >= 0; i--) {
	  int row = mapper->getRow(chan.getHardwareAddress());
	  int col = mapper->getColumn(chan.getHardwareAddress());
	  auto type = mapper->getChannelType(chan.getHardwareAddress());
          std::cout << "Row " << row << " Column " << col << " Type [" << type << "] Timebin " << time << ", ADC " << adcs[i] << std::endl;
          time--;
        }
      }
    }
  }
  return 0;
}
