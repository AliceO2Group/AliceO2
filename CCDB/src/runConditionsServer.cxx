// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "runFairMQDevice.h"

#include "CCDB/ConditionsMQServer.h"

using namespace o2::CDB;
using namespace std;

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()("first-input-name", bpo::value<std::string>()->default_value("first_input.root"),
                        "First input file name")("first-input-type", bpo::value<std::string>()->default_value("ROOT"),
                                                 "First input file type (ROOT/ASCII)")(
    "second-input-name", bpo::value<std::string>()->default_value(""), "Second input file name")(
    "second-input-type", bpo::value<std::string>()->default_value("ROOT"), "Second input file type (ROOT/ASCII)")(
    "output-name", bpo::value<std::string>()->default_value(""), "Output file name")(
    "output-type", bpo::value<std::string>()->default_value("ROOT"), "Output file type")(
    "channel-name", bpo::value<std::string>()->default_value("ROOT"), "Output channel name");
}

FairMQDevice* getDevice(const FairMQProgOptions& config) { return new ConditionsMQServer(); }
