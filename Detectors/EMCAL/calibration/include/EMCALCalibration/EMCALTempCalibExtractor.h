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

/// \class EMCALTempCalibExtractor
/// \brief  Calculate gain correction factors based on the temperature and the cell-by-cell temperature dependent correction factors (slope and intercept)
/// \author Joshua König
/// \ingroup EMCALCalib
/// \since June 30, 2025

#ifndef EMCALTEMPCALIBEXTRACTOR_H_
#define EMCALTEMPCALIBEXTRACTOR_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include "CCDB/BasicCCDBManager.h"
#include "EMCALCalib/ElmbData.h"
#include "EMCALCalib/TempCalibrationParams.h"
#include "EMCALBase/Geometry.h"

namespace o2
{
namespace emcal
{

class EMCALTempCalibExtractor
{

 public:
  /// \brief Constructor
  EMCALTempCalibExtractor()
  {
    LOG(info) << "initialized EMCALTempCalibExtractor";
    try {
      // Try to access geometry initialized ountside
      mGeometry = o2::emcal::Geometry::GetInstance();
    } catch (o2::emcal::GeometryNotInitializedException& e) {
      mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000); // fallback option
    }
  };
  /// \brief Destructor
  ~EMCALTempCalibExtractor() = default;

  /// \brief Initialize temperature data and slope for each cell from the ccdb
  /// \param path path to the slope data
  /// \param timestamp timestamp for the ccdb objects or runnumber (will detect automatically if its a runnumber and convert it)
  void InitializeFromCCDB(std::string path, uint64_t timestamp);

  /// \brief get average temperature in a supermodule
  /// \param iSM SM number
  /// \param ElmbData object where temperature sensor values are stored
  /// \return average temperature in a supermodule
  float getTemperatureForSM(const unsigned short iSM, o2::emcal::ElmbData* ElmbData) const;

  /// \brief get gain calibration factor depending on the temperature and the slope of the cell
  /// \param cellID cell ID
  /// \return gain calibration factor
  float getGainCalibFactor(const unsigned short cellID) const;

  /// \brief set temperature range in which sensor ddata is assumed to be good
  /// \param low lower temperature
  /// \param high upper temperature
  void setAcceptedEnergyRange(float low, float high);

  /// \brief set if median (true) or mean (false) should be used for averaging of the temperature in a SM
  void setUseMedian(const bool tmp) { mUseMedian = tmp; }

  /// \brief get sensor IDs for a specific supermodule
  /// \param iSM SM number
  /// \return vector of sensor IDs
  std::vector<unsigned short> getSensorsForSM(const unsigned short iSM) const;

 private:
  static constexpr unsigned short mNCells = 17664;      ///< Number of EMCal cells
  std::array<float, mNCells> mGainCalibFactors;         ///< gain calibration factors that are calculated based on the temperature and the slopes for each cell
  o2::emcal::Geometry* mGeometry;                       ///< pointer to the EMCal geometry
  std::array<float, 2> mAcceptedTempRange = {15., 30.}; ///< Temperature range where sensors are believed to send good data. Temperatures outside this range will be rejected
  bool mUseMedian = true;                               /// switch to decide if temperature within a SM should be calculated as the mean or the median of the individual sensor data
};

} // namespace emcal

} // namespace o2

#endif