// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/exe/rawdump.cxx
/// \brief  Raw data dumper for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 December 2019

#include <iostream>
#include "fmt/format.h"
#include "boost/program_options.hpp"
#include "DPLUtils/RawParser.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/GBTBareDecoder.h"
#include "MIDRaw/GBTUserLogicDecoder.h"
#include "MIDRaw/RawFileReader.h"

namespace po = boost::program_options;

std::string getOutFilename(const char* inFilename, const char* outDir)
{
  std::string basename(inFilename);
  std::string fdir = "./";
  auto pos = basename.find_last_of("/");
  if (pos != std::string::npos) {
    basename.erase(0, pos + 1);
    fdir = inFilename;
    fdir.erase(pos);
  }
  basename.insert(0, "dump_");
  basename += ".txt";
  std::string outputDir(outDir);
  if (outputDir.empty()) {
    outputDir = fdir;
  }
  if (outputDir.back() != '/') {
    outputDir += "/";
  }
  std::string outFilename = outputDir + basename;
  return outFilename;
}

template <typename GBTDECODER, typename RDH>
void decode(o2::mid::Decoder<GBTDECODER>& decoder, gsl::span<const uint8_t> payload, const RDH& rdh, std::ostream& out)
{
  decoder.clear();
  decoder.process(payload, rdh);
  decoder.flush();
  for (auto& rof : decoder.getROFRecords()) {
    std::stringstream ss;
    ss << std::hex << std::showbase << rof.interactionRecord;
    out << ss.str() << std::endl;
    for (auto colIt = decoder.getData().begin() + rof.firstEntry; colIt != decoder.getData().begin() + rof.firstEntry + rof.nEntries; ++colIt) {
      out << *colIt << std::endl;
    }
  }
}

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");
  std::string outFilename = "", feeIdConfigFilename = "";
  unsigned long int nHBs = 0;
  unsigned long int firstHB = 0;

  // clang-format off
  generic.add_options()
          ("help", "produce help message")
          ("output", po::value<std::string>(&outFilename),"Output text file")
          ("first", po::value<unsigned long int>(&firstHB),"First HB to read")
          ("nHBs", po::value<unsigned long int>(&nHBs),"Number of HBs read")
          ("rdh-only", po::value<bool>()->implicit_value(true),"Only show RDHs")
          ("decode", po::value<bool>()->implicit_value(true),"Decode output")
          ("feeId-config-file", po::value<std::string>(&feeIdConfigFilename),"Filename with crate FEE ID correspondence")
          ("bare", po::value<bool>()->implicit_value(true),"Bare decoder");


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

  bool isBare = (vm.count("bare") > 0);
  bool runDecoder = (vm.count("decode") > 0);

  o2::mid::Decoder<o2::mid::GBTUserLogicDecoder> ulDecoder;
  o2::mid::Decoder<o2::mid::GBTBareDecoder> bareDecoder;

  if (runDecoder) {
    if (!feeIdConfigFilename.empty()) {
      o2::mid::FEEIdConfig feeIdConfig(feeIdConfigFilename.c_str());
      bareDecoder.setFeeIdConfig(feeIdConfig);
      ulDecoder.setFeeIdConfig(feeIdConfig);
    }
    bareDecoder.init(true);
    ulDecoder.init(true);
  }

  std::ofstream outFile;
  std::ostream& out = (outFilename.empty()) ? std::cout : (outFile.open(outFilename), outFile);

  unsigned long int iHB = 0;
  bool isRdhOnly = vm.count("rdh-only") > 0;

  for (auto& filename : inputfiles) {
    // Here we use a custom file reader to be able to read all kind of raw data,
    // even those with a malformed structure in terms of number of HBFs per time frame
    o2::mid::RawFileReader rawFileReader;
    if (!rawFileReader.init(filename.c_str())) {
      return 2;
    }
    while (rawFileReader.readHB()) {
      if (iHB >= firstHB) {
        o2::framework::RawParser parser(rawFileReader.getData().data(), rawFileReader.getData().size());
        auto it = parser.begin(); // We only have 1 HB
        auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
        if (!runDecoder) {
          o2::raw::RDHUtils::printRDH(rdhPtr);
        }
        if (it.size() > 0) {
          gsl::span<const uint8_t> payload(it.data(), it.size());
          if (runDecoder) {
            if (isBare) {
              decode(bareDecoder, payload, *rdhPtr, out);
            } else {
              decode(ulDecoder, payload, *rdhPtr, out);
            }
          } else if (!isRdhOnly) {
            for (size_t iword = 0; iword < payload.size(); iword += 16) {
              auto word = payload.subspan(iword, 16);
              for (auto it = word.rbegin(); it != word.rend(); ++it) {
                auto ibInWord = word.rend() - it;
                if (isBare) {
                  if (ibInWord == 4 || ibInWord == 9) {
                    out << " ";
                  }
                  if (ibInWord == 5 || ibInWord == 10) {
                    out << "  ";
                  }
                }
                out << fmt::format("{:02x}", static_cast<int>(*it));
              }
              out << "\n";
            }
          }
        }
      }
      rawFileReader.clear();
      ++iHB;
      if (nHBs > 0 && iHB >= nHBs + firstHB) {
        break;
      }
    } // loop on HBs
  }   // loop on files
  outFile.close();

  return 0;
}
