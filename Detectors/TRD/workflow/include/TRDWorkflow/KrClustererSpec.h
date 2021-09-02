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

#ifndef O2_TRD_KRCLUSTERERSPEC_H
#define O2_TRD_KRCLUSTERERSPEC_H

/// \file   KrClustererSpec.h
/// \brief Steers the TRD Krypton cluster finder
/// \author Ole Schmidt

// input TRD digits, TRD trigger records
// output Kr clusters

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{
/// create a processor spec
framework::DataProcessorSpec getKrClustererSpec(bool digitTrigRec);

} // namespace trd
} // namespace o2

#endif // O2_TRD_KRCLUSTERERSPEC_H
