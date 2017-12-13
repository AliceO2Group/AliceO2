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
#include "Publishers/DataPublisherDevice.h"

namespace bpo = boost::program_options;
using DataPublisherDevice = o2::utilities::DataPublisherDevice;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (DataPublisherDevice::OptionKeyDataDescription,
     bpo::value<std::string>()->default_value("unspecified"),
     "default data description")
    (DataPublisherDevice::OptionKeyDataOrigin,
     bpo::value<std::string>()->default_value("void"),
     "default data origin")
    (DataPublisherDevice::OptionKeySubspecification,
     bpo::value<DataPublisherDevice::SubSpecificationT>()->default_value(~(DataPublisherDevice::SubSpecificationT)0),
     "default sub specification")
    (DataPublisherDevice::OptionKeyFileName,
     bpo::value<std::string>()->default_value(""),
     "File name")
    (DataPublisherDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel")
    (DataPublisherDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new DataPublisherDevice();
}
