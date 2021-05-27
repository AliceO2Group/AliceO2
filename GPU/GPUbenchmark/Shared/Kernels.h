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
/// \file GPUbenchmark.h
/// \author: mconcas@cern.ch
#ifndef GPUBENCHMARK_H
#define GPUBENCHMARK_H

#include "GPUCommonDef.h"

namespace o2
{
namespace benchmark
{
void hello_util();

class GPUbenchmark final
{
 public:
  GPUbenchmark() = default;
  virtual ~GPUbenchmark() = default;
  void hello();
};

// Steers
void GPUbenchmark::hello()
{
  hello_util();
}
} // namespace benchmark
} // namespace o2
#endif