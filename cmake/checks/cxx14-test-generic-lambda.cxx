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
/// \file cxx14-test-generic-lambda.cxx
/// \brief Generic lambdas check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

auto glambda = [](auto a) { return a; };

int main()
{
  int number = 44;
  return (glambda(number) == number) ? 0 : 1;
}
