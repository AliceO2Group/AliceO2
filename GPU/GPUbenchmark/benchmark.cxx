// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file benchmark.cxx
/// \author: mconcas@cern.ch

#include <iostream>
#include <Kernels.h>

int main()
{
  std::cout << "HELLO WORLD" << std::endl;
  o2::benchmark::GPUbenchmark bm{};
  bm.hello();
  return 0;
}
