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

#include <bitset>
#include <cstdint>
#include <optional>
#include <string_view>
#include "EMCALWorkflow/CalibLoader.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{

namespace emcal
{

class Cell;

/// \class CellRecalibratorSpec
/// \brief Recalibration workflow at cell level
/// \ingroup EMCALworkflow
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Sept 5, 2022
///
/// # Recalibration at cell level
///
/// The workflow recalibrates a cell vector (with corresponding trigger records) for
/// a given timeframe. The following calibrations are supported:
/// - Bad channel calibration
/// - Time calibration
/// - Gain (energy) calibration
/// Calibration objects are handled automatically via the CCDB. Calibrations must be
/// enabled manually and are only applied in case the calibration objects exists in
/// the CCDB.
///
/// While the energy and time calibration only change values within a cell the bad channel
/// calibration decides whether a cell is accepted. Therefore the amount of cells can change
/// for the given trigger. New trigger record objects are created and published to the same
/// subspec as what is used for the output cell vector.
class CellRecalibratorSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param outputspec Subspecification under which the output is posted
  /// \param badChannelCalib If true the bad channel calibration is enabled
  /// \param timeCalib If true the time calibration is enabled
  /// \param gainCalib If true the fain calibration is enabled
  /// \param calibHandler Handler for calibration object loading
  CellRecalibratorSpec(uint32_t outputspec, bool badChannelCalib, bool timeCalib, bool gainCalib, std::shared_ptr<CalibLoader>(calibHandler));

  /// \brief Destructor
  ~CellRecalibratorSpec() final = default;

  /// \brief Initialize recalibrator
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run recalibration of cells for a new timeframe
  /// \param ctx Processing context
  ///
  /// Iterates over all triggers in the timeframe and recalibrates cells. Only
  /// calibrations which are enabled are applied. Calibrated cells and new trigger
  /// records are posted at the end of the timeframe to the subspecification requested
  /// in the constructor using standard types for cells and trigger records
  void run(framework::ProcessingContext& ctx) final;

  /// \brief Fetching cell objects and assigning them to the internal handlers
  /// \param matcher Data type matcher of the CCDB object
  /// \param obj Pointer to CCDB object loaded
  ///
  /// Loading CCDB object into internal cache for the 3 supported CCDB entries (bad
  /// channel map, time calibration params, gain calibration params). Objects are only
  /// loaded in case the calibration type was enabled.
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

  /// \brief Switch for bad channel calibration
  /// \param doRun If true the bad channel calibration is applied
  void setRunBadChannelCalibration(bool doRun) { mCalibrationSettings.set(BADCHANNEL_CALIB, doRun); };

  /// \brief Switch for time calibration
  /// \param doRun If true the time calibration is applied
  void setRunTimeCalibration(bool doRun) { mCalibrationSettings.set(TIME_CALIB, doRun); }

  /// \brief Switch for the gain calibration
  /// \param doRun If true the gain calibration is applied
  void setRunGainCalibration(bool doRun) { mCalibrationSettings.set(GAIN_CALIB, doRun); }

  /// \brief Check if the bad channel calibration is enabled
  /// \return True if the bad channel calibration is enabled, false otherwise
  bool isRunBadChannlCalibration() const { return mCalibrationSettings.test(BADCHANNEL_CALIB); }

  /// \brief Check if the time calibration is enabled
  /// \return True if the time calibration is enabled, false otherwise
  bool isRunTimeCalibration() const { return mCalibrationSettings.test(TIME_CALIB); }

  /// \brief Check if the gain calibration is enabled
  /// \return True if the gain calibration is enabled, false otherwise
  bool isRunGainCalibration() const { return mCalibrationSettings.test(GAIN_CALIB); }

 private:
  /// \brief Apply requested calibrations to the cell
  /// \param inputcell Cell to be calibrated
  /// \return Optional of calibrated cell (empty optional in case the cell is rejected as bad or dead)
  ///
  /// Recalibrating cell for energy and time, and check whether the corresponding tower is not
  /// marked as bad or dead. Only calibrations which are enabled are applied.
  std::optional<o2::emcal::Cell> getCalibratedCell(const o2::emcal::Cell& inputcell) const;

  /// \brief Update internal cache of calibration objects
  void updateCalibObjects();

  /// \enum CalibrationType_t
  /// \brief Calibrations handled by the recalibration workflow
  enum CalibrationType_t {
    BADCHANNEL_CALIB = 0, ///< Bad channel calibration
    TIME_CALIB = 1,       ///< Time calibration
    GAIN_CALIB = 2        ///< Gain calibration
  };

  uint32_t mOutputSubspec = 0;                              ///< output subspecification;
  std::bitset<8> mCalibrationSettings;                      ///< Recalibration settings (which calibration to be applied)
  std::shared_ptr<CalibLoader> mCalibrationHandler;         ///< Handler loading calibration objects
  const BadChannelMap* mBadChannelMap = nullptr;            ///< Bad channelMap
  const TimeCalibrationParams* mTimeCalibration = nullptr;  ///< Time calibration coefficients
  const GainCalibrationFactors* mGainCalibration = nullptr; ///< Gain calibration factors
};

/// \brief Create CellRecalibrator processor spec
/// \param inputSubsepc Subspecification used for the input objects (cells and trigger records)
/// \param outputSubspec Subspecification used for the output objects (cells and trigger records)
/// \param badChannelCalib If true the bad channel calibration is enabled
/// \param timeCalib If true the time calibration is enabled
/// \param gainCalib If true the gain (energy) calibration is enabled
framework::DataProcessorSpec getCellRecalibratorSpec(uint32_t inputSubspec, uint32_t outputSubspec, bool badChannelCalib, bool timeCalib, bool gainCalib, const std::string_view pathBadChannelMap, const std::string_view pathTimeCalib, std::string_view pathGainCalib);

} // namespace emcal

} // namespace o2