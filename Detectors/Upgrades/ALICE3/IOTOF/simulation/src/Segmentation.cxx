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

/// \file Segmentation.cxx
/// \brief Implementation of the Segmentation class

#include "IOTOFSimulation/Segmentation.h"
#include "IOTOFBase/IOTOFBaseParam.h"
#include <cstdio>

namespace o2
{

namespace iotof
{

std::unique_ptr<o2::iotof::Segmentation> Segmentation::sInstance;

Segmentation* Segmentation::Instance()
{
  if (!sInstance) {
    sInstance = std::unique_ptr<Segmentation>(new Segmentation());
  }
  return sInstance.get();
}

Segmentation::Segmentation()
{
  if (sInstance) {
    printf("Invalid use of public constructor: o2::iotof::Segmentation instance exists\n");
  }
}

} // namespace iotof
} // namespace o2

ClassImp(o2::iotof::Segmentation);
