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
/// \file   make-parameter-collection.cxx
/// \author Nicolò Jacazio nicolo.jacazio@cern.ch
/// \since  2022-11-08
/// \brief  A simple tool to produce resolution parametrization object parameters
///

#include "TGraph.h"
#include "TSystem.h"
#include "TCanvas.h"
#include <boost/program_options.hpp>
#include "TFile.h"
#include "DataFormatsTOF/ParameterContainers.h"

namespace bpo = boost::program_options;
using namespace std::chrono;
using namespace o2::tof;

bpo::variables_map arguments; // Command line arguments
bool initOptionsAndParse(bpo::options_description& options, int argc, char* argv[])
{
  options.add_options()(
    "ccdb-path,c", bpo::value<std::string>()->default_value("Analysis/PID/TOF"), "CCDB path for storage/retrieval")(
    "reso-name,n", bpo::value<std::string>()->default_value("TOFResoParams"), "Name of the parametrization object")(
    "mode,m", bpo::value<unsigned int>()->default_value(1), "Working mode: 0 push, 1 pull and test, 2 create and performance")(
    "p0", bpo::value<float>()->default_value(0.008f), "Parameter 0 of the TOF resolution")(
    "p1", bpo::value<float>()->default_value(0.008f), "Parameter 1 of the TOF resolution")(
    "p2", bpo::value<float>()->default_value(0.002f), "Parameter 2 of the TOF resolution")(
    "p3", bpo::value<float>()->default_value(40.0f), "Parameter 3 of the TOF resolution")(
    "p4", bpo::value<float>()->default_value(60.0f), "Parameter 4 of the TOF resolution: average TOF resolution");
  try {
    bpo::store(parse_command_line(argc, argv, options), arguments);

    // help
    if (arguments.count("help")) {
      LOG(info) << options;
      return false;
    }

    bpo::notify(arguments);
  } catch (const bpo::error& e) {
    LOG(error) << e.what() << "\n";
    LOG(error) << "Error parsing command line arguments; Available options:";
    LOG(error) << options;
    return false;
  }
  return true;
}

class ParamExample : public Parameters<5>
{
 public:
  ParamExample() : Parameters(std::array<std::string, 5>{"p0", "p1", "p2", "p3", "p4"},
                              "ParamExample"){}; // Default constructor with default parameters
  ~ParamExample() = default;
};

int main(int argc, char* argv[])
{
  bpo::options_description options("Allowed options");
  if (!initOptionsAndParse(options, argc, argv)) {
    LOG(info) << "Nothing to do";
    return 1;
  }
  ParamExample parameters;

  parameters.print();

  return 0;
}
