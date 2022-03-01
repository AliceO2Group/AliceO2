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

/// \file DsChannleGroup.h
/// \brief Implementation of a group of DualSampa channels
///
/// \author Andrea Ferrero, CEA-Saclay

#ifndef ALICEO2_MCH_DSCHANNELGROUP_H_
#define ALICEO2_MCH_DSCHANNELGROUP_H_

#include <vector>
#include "DataFormatsMCH/DsChannelId.h"

namespace o2::mch
{

using DsChannelGroup = std::vector<DsChannelId>;

} // namespace o2::mch

#endif
