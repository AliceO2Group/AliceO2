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
/// \file cxx14-test-make_unique.cxx
/// \brief std::make_unique check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

#include <memory>

bool checkPointer(int number)
{
  auto pointer = std::make_unique<int>(number);
  return *pointer == number;
}

int main()
{
  bool ret = checkPointer(41);
  return ret ? 0 : 1;
}
