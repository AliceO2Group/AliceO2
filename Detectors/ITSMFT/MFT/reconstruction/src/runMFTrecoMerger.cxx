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

#include "MFTReconstruction/devices/Merger.h"

using namespace o2::MFT;

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{

}

FairMQDevicePtr getDevice(const FairMQProgOptions& config)
{
  return new Merger();
}
