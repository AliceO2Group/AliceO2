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

/// \file rawReaderFileNew.cxx
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory

#include <iostream>
#include <boost/program_options.hpp>

#include <TTree.h>

#include "DetectorsRaw/RawFileReader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALBase/Mapper.h"
#include "EMCALBase/TriggerMappingV2.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/RawReaderMemory.h"
#include <fairlogger/Logger.h>

namespace bpo = boost::program_options;
// using namespace o2::emcal;

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " <cmds/options>\n"
                                       "  Tool will decode the DDLx data for EMCAL 0\n"
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
      // std::cout << GitInfo();
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
  reader.setDefaultDataOrigin(o2::header::gDataOriginEMC);
  reader.setDefaultDataDescription(o2::header::gDataDescriptionRawData);
  reader.setDefaultReadoutCardType(o2::raw::RawFileReader::RORC);
  reader.addFile(rawfilename);
  reader.init();

  o2::emcal::MappingHandler mapper;
  o2::emcal::TriggerMappingV2 triggermapping;

  std::unique_ptr<TFile> treefile(TFile::Open("trudata.root", "RECREATE"));
  TTree trudata("trudata", "Tree with TRU data");
  // branches in tree
  struct collisiontrigger {
    unsigned long bc;
    unsigned long orbit;
  } mycollision;
  int absFastOR;
  int starttime;
  std::vector<int> timesamples;
  tree->Branch(&mycollision, "collisiontrigger", "bc,orbit/l");
  tree->Branch(&starttime, "starttime", "starttime/i");
  tree->Branch(&timesamples, "timesamples", ""); // @todo check how to write std::vector to tree;

  while (1) {
    int tfID = reader.getNextTFToRead();
    if (tfID >= reader.getNTimeFrames()) {
      LOG(info) << "nothing left to read after " << tfID << " TFs read";
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
      o2::emcal::RawReaderMemory parser(dataBuffer);
      while (parser.hasNext()) {
        parser.next();
        auto rdh = parser.getRawHeader();
        auto ddl = o2::raw::RDHUtils::getFEEID(parser.getRawHeader());
        // Exclude STU DDLs
        if (ddl >= 40) {
          continue;
        }

        mycollision.bc = o2::raw::RDHUtils::getTriggerBC(rdh);
        mycollision.orbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);

        o2::emcal::AltroDecoder decoder(parser);
        decoder.decode();
        auto& ddlmapping = mapper.getMappingForDDL(ddl);

        std::cout << decoder.getRCUTrailer() << std::endl;
        for (auto& chan : decoder.getChannels()) {
          if (ddlmapping.getChannelType(chan.getHardwareAddress) != o2::emcal::ChannelType_t::TRU) {
            continue;
          }
          std::cout << "Hw address: " << chan.getHardwareAddress() << std::endl;
          // Get absolute FastOR index - this will tell us where on the EMCAL surface the FastOR is
          // TRU index is encoded in column, needs to be converted to an absoluted FastOR ID via the
          // trigger mapping. The absoluted FastOR ID can be connected via the geometry to tower IDs
          // from the FEC data.
          // we are only interested in the FastORs for now, skip patches starting from 96
          auto fastorInTRU = ddlmapping.getColumn(chan.getHardwareAddress());
          if (fastorInTRU >= 96) {
            // indices starting from 96 encode patches, not FastORs
            continue;
          }
          auto truindex = triggermapping.getTRUIndexFromOnlineHardareAddree(chan.getHardwareAddress(), ddl, ddl / 2);
          auto absFastOrID = triggermapping.getAbsFastORIndexFromIndexInTRU(truindex, fastorInTRU);

          for (auto& bunch : chan.getBunches()) {
            std::cout << "BunchLength: " << int(bunch.getBunchLength()) << std::endl;
            auto adcs = bunch.getADC();
            int time = bunch.getStartTime();
            starttime = time;
            timesamples.clear();
            timesamples.resize(adcs.size());
            std::copy(adcs.begin(), adcs.end(), timesamples.begin());
            trudata.Fill();
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