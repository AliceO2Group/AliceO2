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
#include "DataFormatsEMCAL/MCLabel.h"
#include "EMCALCalib/CellRecalibrator.h"
#include "EMCALWorkflow/CalibLoader.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{

namespace emcal
{

class Cell;
class TriggerRecord;

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
  /// \enum LEDEventSettings
  /// \brief Dedicated handling for LED events
  enum class LEDEventSettings {
    KEEP,    ///< Keep LED events in timeframe (uncalibrated)
    DROP,    ///< Drop LED events
    REDIRECT ///< Redirect LED events to dedicated output
  };
  /// \brief Constructor
  /// \param outputspec Subspecification under which the output is posted
  /// \param ledsettings Handling of cells from LED events
  /// \param badChannelCalib If true the bad channel calibration is enabled
  /// \param timeCalib If true the time calibration is enabled
  /// \param gainCalib If true the gain calibration is enabled
  /// \param isMC If true the MCLabelContainer is adapted
  /// \param calibHandler Handler for calibration object loading
  CellRecalibratorSpec(uint32_t outputspec, LEDEventSettings ledsettings, bool badChannelCalib, bool timeCalib, bool gainCalib, bool isMC, std::shared_ptr<CalibLoader>(calibHandler));

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
  /// \brief Update calibration objects (if changed)
  void updateCalibObjects();

  /// \brief write event cell container to output
  /// \param selectedCells Cells to be added
  /// \param outputcontainer Output container
  /// \param outputtriggers Output trigger records
  void writeTrigger(const gsl::span<const o2::emcal::Cell> selectedCells, const o2::emcal::TriggerRecord& eventtrigger, std::vector<o2::emcal::Cell>& outputcontainer, std::vector<o2::emcal::TriggerRecord>& outputtriggers);

  void writeMCLabels(const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>& inputlabels, o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>& outputContainer, const std::vector<int>& keptIndices, int firstindex);

  /// \enum CalibrationType_t
  /// \brief Calibrations handled by the recalibration workflow
  enum CalibrationType_t {
    BADCHANNEL_CALIB = 0, ///< Bad channel calibration
    TIME_CALIB = 1,       ///< Time calibration
    GAIN_CALIB = 2        ///< Gain calibration
  };

  uint32_t mOutputSubspec = 0;                            ///< output subspecification;
  bool mIsMC = false;                                     ///< MC mode
  LEDEventSettings mLEDsettings = LEDEventSettings::KEEP; ///< Handling of LED events
  std::bitset<8> mCalibrationSettings;                    ///< Recalibration settings (which calibration to be applied)
  std::shared_ptr<CalibLoader> mCalibrationHandler;       ///< Handler loading calibration objects
  CellRecalibrator mCellRecalibrator;                     ///< Recalibrator at cell level
};

/// \brief Create CellRecalibrator processor spec
/// \param inputSubsepc Subspecification used for the input objects (cells and trigger records)
/// \param outputSubspec Subspecification used for the output objects (cells and trigger records)
/// \param ledsettings Settings of LED handling (keep/drop/redirect)
/// \param badChannelCalib If true the bad channel calibration is enabled
/// \param timeCalib If true the time calibration is enabled
/// \param gainCalib If true the gain (energy) calibration is enabled
/// \param isMC If true also the MC label container is adapted (relevant only for bad channel masking)
framework::DataProcessorSpec getCellRecalibratorSpec(uint32_t inputSubspec, uint32_t outputSubspec, uint32_t ledsettings, bool badChannelCalib, bool timeCalib, bool gainCalib, bool isMC);

} // namespace emcal

} // namespace o2