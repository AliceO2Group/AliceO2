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

#include "CCDB/ConditionsMQClient.h"

using namespace o2::CDB;
using namespace std;

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()("parameter-name", bpo::value<string>()->default_value("DET/Calib/Histo"), "Parameter Name")(
    "operation-type", bpo::value<string>()->default_value("GET"), "Operation Type")(
    "data-source", bpo::value<string>()->default_value("OCDB"), "Data Source")(
    "object-path", bpo::value<string>()->default_value("OCDB"), "Object Path");
}

FairMQDevice* getDevice(const FairMQProgOptions& config) { return new ConditionsMQClient(); }
