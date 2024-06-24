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

#ifndef SIMDLLOADER_H_
#define SIMDLLOADER_H_

#include "CommonUtils/DLLoaderBase.h"

namespace o2::conf
{

class SimDLLoader : public o2::utils::DLLoaderBase<SimDLLoader>
{
  O2DLLoaderDef(SimDLLoader)
};

} // namespace o2::conf

#endif // SIMDLLOADER_H_
