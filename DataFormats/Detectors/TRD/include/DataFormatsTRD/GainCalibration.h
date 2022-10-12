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

/// \file GainCalibration.h
/// \brief This file holds the gain calibration object
/// \author Felix Schlepper

#ifndef O2_TRD_GAINCALIBRATION_H
#define O2_TRD_GAINCALIBRATION_H

#include <gsl/span>

namespace o2
{
namespace trd
{

/// Defines the gain calibration object.
/// ...
class GainCalibration
{
 public:
  /// Default constructor
  GainCalibration() = default;
  /// Default copy constructor
  GainCalibration(const GainCalibration&) = default;
  /// Default destructor
  ~GainCalibration() = default;

  // TODO
  void fill(const GainCalibration& input) {}
  void fill(const gsl::span<const GainCalibration> input) {}
  void merge(const GainCalibration* prev) {}
  void print() {}

 private:
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_GAINCALIBRATION_H
