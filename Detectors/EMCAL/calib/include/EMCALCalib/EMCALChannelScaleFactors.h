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

/// \class EMCALChannelScaleFactors
/// \brief  Container for energy dependent scale factors for number of hits in a cell
/// \author Joshua Koenig
/// \ingroup EMCALCalib
/// \since May 13, 2022

#ifndef EMCAL_CHANNEL_SCALE_FACTORS_H_
#define EMCAL_CHANNEL_SCALE_FACTORS_H_

#include <cfloat>
#include "Framework/Logger.h"
#include "EMCALBase/Geometry.h"

namespace o2
{
namespace emcal
{

class InvalidEnergyIntervalException final : public std::exception
{

 public:
  /// \brief Constructor
  /// \param E Cell energy requested
  InvalidEnergyIntervalException(float E, int cellID) : std::exception(),
                                                        mE(E),
                                                        mCellID(cellID),
                                                        mMessage("Invalid energy " + std::to_string(mE) + "for cell " + std::to_string(mCellID) + "]")
  {
  }

  /// \brief Destructor
  ~InvalidEnergyIntervalException() noexcept final = default;

  /// \brief Access to error message
  /// \return Error message for given exception
  const char* what() const noexcept final { return mMessage.c_str(); }

 private:
  float mE;             ///< Cell energy requested
  int mCellID;          ///< Cell ID
  std::string mMessage; ///< Message to be printed
};

class EnergyIntervals
{
 public:
  EnergyIntervals() = default;
  EnergyIntervals(float min, float max)
  {
    Elow = min;
    Ehigh = max;
  }
  ~EnergyIntervals() = default;

  /// Set the energy interval
  /// \param min The lower bound of the energy interval
  /// \param max The upper bound of the energy interval
  void setEnergy(float min, float max)
  {
    Elow = min;
    Ehigh = max;
  }
  /// Check if the energy is in the energy interval
  /// \param energy The energy to check
  /// \return True if the energy is in the energy interval
  bool isInInterval(float E) const
  {
    LOG(debug) << "EMCALChannelScaleFactors::EnergyIntervals::IsInInterval: Checking if " << E << " is in the interval " << Elow << " - " << Ehigh;
    if (E >= Elow && E < Ehigh) {
      return true;
    } else {
      return false;
    }
  }

  /// declare operator ==, !=, <, >
  bool operator==(const EnergyIntervals& other) const
  {
    if (std::abs(Elow - other.Elow) < FLT_EPSILON && std::abs(Ehigh - other.Ehigh) < FLT_EPSILON) {
      return true;
    } else {
      return false;
    }
  }
  bool operator!=(const EnergyIntervals& other) const
  {
    if (std::abs(Elow - other.Elow) < FLT_EPSILON && std::abs(Ehigh - other.Ehigh) < FLT_EPSILON) {
      return false;
    } else {
      return true;
    }
  }
  bool operator<(const EnergyIntervals& other) const
  {
    if (Elow < other.Elow) {
      return true;
    } else {
      return false;
    }
  }
  bool operator>(const EnergyIntervals& other) const
  {
    if (Elow > other.Elow) {
      return true;
    } else {
      return false;
    }
  }

 private:
  /// declare energy upper and lower bound
  float Elow;
  float Ehigh;

  ClassDefNV(EnergyIntervals, 1);
};

class EMCALChannelScaleFactors
{

 public:
  /// Insert value into the map
  /// \param cell The cell number
  /// \param E_min Minimum energy of the interval
  /// \param E_max Maximum energy of the interval
  /// \param scale Scale factor for number of hits in bad channel calibration
  void insertVal(const int cellID, float E_min, float E_max, float scale);

  /// Get the scale factor for a given cell and energy
  /// \param cell The cell number
  /// \param E The energy
  /// \return The scale factor
  float getScaleVal(const int cellID, float E) const;

 private:
  static constexpr int NCells = 17664;                               ///< Number of cells in the EMCal
  std::array<std::map<EnergyIntervals, float>, NCells> ScaleFactors; ///< Scale factors for each cell and energy interval

  ClassDefNV(EMCALChannelScaleFactors, 1);
};

} // end namespace emcal

} // end namespace o2

#endif
