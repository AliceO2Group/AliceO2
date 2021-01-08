// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file rawReaderFileNew.cxx
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory

#include <iostream>
#include <boost/program_options.hpp>

#include "DetectorsRaw/RawFileReader.h"
#include "PHOSReconstruction/AltroDecoder.h"
#include "PHOSReconstruction/RawReaderMemory.h"
#include "FairLogger.h"

namespace bpo = boost::program_options;
//using namespace o2::phos;

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " <cmds/options>\n"
                                       "  Tool will decode the DDLx data for PHOS 0\n"
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

  auto rawfilename = vm["input-file"].as<std::string>();

  o2::raw::RawFileReader reader;
  reader.setDefaultDataOrigin(o2::header::gDataOriginPHS);
  reader.setDefaultDataDescription(o2::header::gDataDescriptionRawData);
  reader.setDefaultReadoutCardType(o2::raw::RawFileReader::RORC);
  reader.addFile(rawfilename);
  reader.init();

  while (1) {
    int tfID = reader.getNextTFToRead();
    if (tfID >= reader.getNTimeFrames()) {
      LOG(INFO) << "nothing left to read after " << tfID << " TFs read";
      break;
    }
    std::vector<char> dataBuffer; // where to put extracted data
    for (int il = 0; il < reader.getNLinks(); il++) {
      auto& link = reader.getLink(il);
      std::cout << "Decoding link " << il << std::endl;

      auto sz = link.getNextTFSize(); // size in bytes needed for the next TF of this link
      dataBuffer.resize(sz);
      link.readNextTF(dataBuffer.data());

      // Parse
      o2::phos::RawReaderMemory parser(dataBuffer);
      while (parser.hasNext()) {
        parser.next();
        // Exclude STU DDLs
        if (o2::raw::RDHUtils::getFEEID(parser.getRawHeader()) >= 40) {
          continue;
        }
        o2::phos::AltroDecoder decoder(parser);
        decoder.decode();

        auto& rcu = decoder.getRCUTrailer();
        auto& channellist = decoder.getChannels();
        std::cout << rcu << std::endl;
        for (auto& chan : channellist) {
          std::cout << "Hw address: " << chan.getHardwareAddress() << std::endl;
          for (auto& bunch : chan.getBunches()) {
            std::cout << "BunchLength: " << int(bunch.getBunchLength()) << std::endl;
            auto adcs = bunch.getADC();
            int time = bunch.getStartTime();
            for (int i = adcs.size() - 1; i >= 0; i--) {
              std::cout << "Timebin " << time << ", ADC " << adcs[i] << std::endl;
              time--;
            }
          }
        }
      }
    }
    reader.setNextTFToRead(++tfID);
  }
}
