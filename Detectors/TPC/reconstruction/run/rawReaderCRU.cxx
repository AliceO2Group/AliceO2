// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file rawReaderCRU.cxx
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)
/// \author Torsten Alt (Torsten.Alt@cern.ch)

#include <boost/program_options.hpp>

#include "TPCReconstruction/RawReaderCRU.h"

namespace bpo = boost::program_options;
using namespace o2::tpc;
using rawreader::RawReaderCRU;

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
    add_option("output-file,o", bpo::value<std::string>(), "Specify output file prefix (defaults to (dirname+basename) of input-file)");
    add_option("timebins,t", bpo::value<uint32_t>()->default_value(0), "Number of timebins to decode [0 = complete file]");
    add_option("debug,d", bpo::value<uint32_t>()->default_value(0), "Select debug output level [0 = no debug output]");
    add_option("stream,s", bpo::value<uint32_t>()->default_value(0), "Stream to decode [default = 0]");
    add_option("link,l", bpo::value<uint32_t>()->default_value(0), "GBT link to decode [default = 0]");
    add_option("json", "Output results as json (if applicable)");

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

  RawReaderCRU::processFile(
    vm["input-file"].as<std::string>(),
    vm["timebins"].as<uint32_t>(),
    vm["link"].as<uint32_t>(),
    vm["stream"].as<uint32_t>(),
    vm["debug"].as<uint32_t>(),
    vm["verbose"].as<uint32_t>(),
    vm.count("output-file") ? vm["output-file"].as<std::string>() : "");

  return 0;
}
