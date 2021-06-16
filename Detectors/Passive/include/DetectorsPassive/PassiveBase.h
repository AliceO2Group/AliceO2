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

#ifndef ALICEO2_PASSIVE_BASE_H
#define ALICEO2_PASSIVE_BASE_H

#include "FairModule.h" // for FairModule

namespace o2
{
namespace passive
{

/// a common base class for passive modules - implementing generic functions
class PassiveBase : public FairModule
{
 public:
  using FairModule::FairModule;
  void SetSpecialPhysicsCuts() override;

  ClassDefOverride(o2::passive::PassiveBase, 1);
};

} // namespace passive
} // namespace o2

#endif
