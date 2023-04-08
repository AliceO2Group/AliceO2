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

#include <catch_amalgamated.hpp>
#include "Framework/Logger.h"

TEST_CASE("TestLogF")
{
  LOGF(error, "%s", "Hello world");
  LOGF(info, "%s", "Hello world");
  LOGF(error, "%.2f", 1000.30343f);
  LOGP(error, "{}", "Hello world");
  LOGP(info, "{}", "Hello world");
  LOGP(error, "{:03.2f}", 1000.30343f);
  LOGP(error, "{1} {0}", "world", "Hello");
  O2ERROR("{}", "Hello world");
  O2INFO("{}", "Hello world");
}
