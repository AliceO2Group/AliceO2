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

/// \file CalGain.h
/// \brief This file holds the gain calibration object to be written into the CCDB
/// \author Felix Schlepper

#ifndef O2_TRD_CALGAIN_H
#define O2_TRD_CALGAIN_H

namespace o2
{
namespace trd
{

/// Defines the gain calibration object.
/// ...
class CalGain
{
 public:
  /// Default constructor
  CalGain() = default;
  /// Default copy constructor
  CalGain(const CalGain&) = default;
  /// Default destructor
  ~CalGain() = default;

 private:
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CALGAIN_H
