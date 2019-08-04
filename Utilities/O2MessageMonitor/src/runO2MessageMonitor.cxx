// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

/// @file runO2MessageMonitor.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#include "runFairMQDevice.h"
#include "O2MessageMonitor/O2MessageMonitor.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  // clang-format off
  options.add_options()
    ("n",bpo::value<int>()->default_value(-1), "How many loops");
  options.add_options()
    ("sleep",bpo::value<int>()->default_value(0), "sleep between loops in milliseconds");
  options.add_options()
    ("limit",bpo::value<int>()->default_value(0), "limit output of payload to n characters");
  options.add_options()
    ("payload",bpo::value<std::string>()->
     default_value("I am the info payload"), "the info string in the payload");
  options.add_options()
    ("name",bpo::value<std::string>()->default_value(""), "optional name in the header");
  // clang-format on
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
    return new O2MessageMonitor();
}
