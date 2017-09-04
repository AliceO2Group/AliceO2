// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "runFairMQDevice.h"

#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/FindHits.h"
#include "MFTReconstruction/devices/Sampler.h"

using namespace o2::MFT;

namespace bpo = boost::program_options;

//_____________________________________________________________________________
void addCustomOptions(bpo::options_description& options)
{

  options.add_options()
    ("file-name",bpo::value<std::vector<std::string>>(),"Path to the input file")
    ("max-index",bpo::value<int64_t>()->default_value(-1),"number of events to read")
    ("branch-name",bpo::value<std::vector<std::string>>()->required(),"branch name")
    ("out-channel",bpo::value<std::string>()->default_value("data-out"),"output channel name")
    ("ack-channel",bpo::value<std::string>()->default_value(""),"ack channel name");

}

//_____________________________________________________________________________
FairMQDevicePtr getDevice(const FairMQProgOptions& config)
{

  std::vector<std::string> filename = config.GetValue<std::vector<std::string>>("file-name");
  std::vector<std::string> branchname = config.GetValue<std::vector<std::string>>("branch-name");

  auto* sampler = new Sampler();

  for (UInt_t ielem = 0; ielem < filename.size(); ielem++) {
    sampler->addInputFileName(filename.at(ielem));
  }
  
  for (UInt_t ielem = 0; ielem < branchname.size(); ielem++) {
    sampler->addInputBranchName(branchname.at(ielem));
    LOG(INFO) << "Run::Sampler >>>>> add input branch " << branchname.at(ielem).c_str() << "";
  }

  return sampler;

}
