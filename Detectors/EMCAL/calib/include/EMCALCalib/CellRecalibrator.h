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
#ifndef ALCEO2_EMCAL_CELLRECALIBRATOR_H
#define ALCEO2_EMCAL_CELLRECALIBRATOR_H

#include <exception>
#include <iosfwd>
#include <optional>
#include <vector>
#include <tuple>
#include <gsl/span>

#include "Rtypes.h"

#include <DataFormatsEMCAL/Constants.h>
#include <EMCALCalib/BadChannelMap.h>
#include <EMCALCalib/TimeCalibrationParams.h>
#include <EMCALCalib/GainCalibrationFactors.h>

namespace o2
{

namespace emcal
{

/// \class CellRecalibrator
/// \brief Tool for recalibration at cell level
/// \ingroup EMCALcalib
/// \author Markus Fasel <markus.fasel@cern.ch> Oak Ridge National Laboratory
/// \since Oct 12, 2022
///
/// Applying cell-level calibrations
/// - bad channel removal
/// - time shift
/// - gain calibration
///
/// Attention: All calibrations for which calibration objects
/// are provided are applied. Check for active calibration is
/// therefore related to the presence of the corresponding
/// calibration object.
///
/// Input can be a single cell (getCalibratedCell) or a
/// collection of cells (getCalibratedCells). In case of
/// single cell the result is optional since the cell can
/// be rejected by the bad channel mask. Only cells of type
/// high gain or low gain can be calibrated, in case cells
/// of other types (LEDMON/TRU) are passed to getCalibratedCell
/// an exception will be thrown.
///
/// The calibrator supports all cells of the CellInterface
/// concept.
class CellRecalibrator
{
 public:
  /// \class CellTypeException
  /// \brief Handling of invalid cell types in calibration
  class CellTypeException : public std::exception
  {
   public:
    /// \brief Constructor
    CellTypeException() = default;

    /// \brief Destructor
    ~CellTypeException() noexcept final = default;

    /// \brief Get error message of the exception
    /// \return Error message
    const char* what() const noexcept final
    {
      return "Only possible to calibrate cells of type high gain or low gain";
    }
  };

  /// \brief Constructor
  CellRecalibrator() = default;

  /// \brief Destructor
  ~CellRecalibrator() = default;

  /// \brief Set the bad channel map
  /// \param bcm Bad channel map to be applied
  void setBadChannelMap(const BadChannelMap* bcm) { mBadChannelMap = bcm; }

  /// \brief Set the time calibration params
  /// \param tcp Time calibration params to be applied
  void setTimeCalibration(const TimeCalibrationParams* tcp) { mTimeCalibration = tcp; }

  /// \brief Set the gain calibration params
  /// \param gcf Gain calibration factors to be applied
  void setGainCalibration(const GainCalibrationFactors* gcf) { mGainCalibration = gcf; }

  /// \brief Check if the bad channel calibration is enabled
  /// \return True if the bad channel calibration is active (object available), false otherwise
  bool hasBadChannelMap() const { return mBadChannelMap != nullptr; }

  /// \brief Check if the time calibration is enabled
  /// \return True if the time calibration is active (object available), false otherwise
  bool hasTimeCalibration() const { return mTimeCalibration != nullptr; }

  /// \brief Check if the energy calibration is enabled
  /// \return True if the energy calibration is active (object available), false otherwise
  bool hasGainCalibration() const { return mGainCalibration != nullptr; }

  /// \brief Get bad channel map currently used in the calibrator
  /// \return Current bad channel map (nullptr if not set)
  const BadChannelMap* getBadChannelMap() const { return mBadChannelMap; }

  /// \brief Get time calibration parameters currently used in the calibrator
  /// \return Current time calibration parameters (nullptr if not set)
  const TimeCalibrationParams* getTimeCalibration() const { return mTimeCalibration; }

  /// \brief Get gain calibration factors currently used in the calibrator
  /// \return Current gain calibration factors (nullptr if not set)
  const GainCalibrationFactors* getGainCalibration() const { return mGainCalibration; }

  /// \brief Calibrate single cell
  ///
  /// Applying all calibrations provided based on presence of calibration objects. Only
  /// cells of type high-gain or low-gain can be calibrated. Since the cell can be rejected
  /// by the bad channel calibration the return type is std::optional, where an empty optional
  /// reflects rejected cells.
  ///
  /// \param inputcell Cell to calibrate
  /// \return Calibrated cell (empty optional if rejected by the bad channel map)
  /// \throw CellTypeException in case the inputcell is neither of type high gain nor low gain
  template <typename T>
  std::optional<T> getCalibratedCell(const T& inputcell) const
  {
    if (!(inputcell.getHighGain() || inputcell.getLowGain())) {
      throw CellTypeException();
    }
    if (hasBadChannelMap()) {
      if (mBadChannelMap->getChannelStatus(inputcell.getTower()) != BadChannelMap::MaskType_t::GOOD_CELL) {
        return std::optional<T>();
      }
    }

    float calibratedEnergy = inputcell.getEnergy();
    float calibratedTime = inputcell.getTimeStamp();

    if (hasTimeCalibration()) {
      calibratedTime -= mTimeCalibration->getTimeCalibParam(inputcell.getTower(), inputcell.getLowGain());
    }

    if (hasGainCalibration()) {
      calibratedEnergy *= mGainCalibration->getGainCalibFactors(inputcell.getTower());
    }
    return std::make_optional<T>(inputcell.getTower(), calibratedEnergy, calibratedTime, inputcell.getHighGain() ? ChannelType_t::HIGH_GAIN : ChannelType_t::LOW_GAIN);
  }

  /// \brief Get list of calibrated cells based on a cell input collection
  ///
  /// Applying calibrations to all cells in the input collections and return
  /// a vector of accepted and calibrated cells. Cells not of type high gain
  /// or low gain are discarded. All calibrations for which calibration objects
  /// are available are applied.
  ///
  /// \param inputcells Collection of input cells
  /// \return Tuple with vector of calibrated cells and vector of kept indices.
  template <typename T>
  std::tuple<std::vector<T>, std::vector<int>> getCalibratedCells(const gsl::span<const T> inputcells)
  {
    std::vector<T> result;
    std::vector<int> indices;
    int currentindex = 0;
    for (const auto& cellToCalibrate : inputcells) {
      if (!(cellToCalibrate.getHighGain() || cellToCalibrate.getLowGain())) {
        currentindex++;
        continue;
      }
      auto calibrated = getCalibratedCell(cellToCalibrate);
      if (calibrated) {
        result.push_back(calibrated.value());
        indices.emplace_back(currentindex);
      }
      currentindex++;
    }
    return std::make_tuple(result, indices);
  }

  /// \brief Print settings to the stream
  /// \param stream Stream to print on
  void printStream(std::ostream& stream) const;

 private:
  const BadChannelMap* mBadChannelMap = nullptr;            ///< Bad channel map
  const TimeCalibrationParams* mTimeCalibration = nullptr;  ///< Time calibration parameters
  const GainCalibrationFactors* mGainCalibration = nullptr; ///< Gain calibration parameters

  ClassDefNV(CellRecalibrator, 1);
};

/// \brief Output stream operator for cell-level calibrator
/// \param in Stream to print on
/// \param calib CellRecalibrator object to be printed
/// \return Stream after printing
std::ostream& operator<<(std::ostream& in, const CellRecalibrator& calib);

} // namespace emcal

} // namespace o2

#endif // !ALCEO2_EMCAL_CELLRECALIBRATOR_H