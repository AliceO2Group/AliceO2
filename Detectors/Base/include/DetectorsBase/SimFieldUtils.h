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

/// \file SimFieldUtils.h
/// \brief Methods to create simulation mag field

#ifndef ALICEO2_BASE_SIMFIELDUTILS_H_
#define ALICEO2_BASE_SIMFIELDUTILS_H_

class FairField;

namespace o2
{
// namespace field

//{
// class MagneticField;
//}

namespace base
{

class SimFieldUtils
{
 public:
  // a common entry point to create the mag field for simulation
  // based on the simulation configuration in SimConfig
  static FairField* const createMagField();
};

} // namespace base
} // namespace o2
#endif
