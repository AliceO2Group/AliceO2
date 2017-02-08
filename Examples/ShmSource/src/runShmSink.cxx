/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @file runShmSink.h
///
/// @since 2017-01-01
/// @author M. Krzewicki <mkrzewic@cern.ch>

#include "runFairMQDevice.h"
#include "ShmSource/ShmSink.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    ("n", bpo::value<int>()->default_value(1000000), "how many messages to send");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
    return new ShmSink();
}
