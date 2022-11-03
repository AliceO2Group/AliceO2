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
#include <vector>
#include <Framework/ProcessingContext.h>
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InputSpec.h"

namespace o2
{

namespace emcal
{
class BadChannelMap;
class TimeCalibrationParams;
class GainCalibrationFactors;

/// \class CalibLoader
/// \brief Handler for EMCAL calibration objects in DPL workflows
/// \ingroup EMCALWorkflow
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Sept. 5, 2022
///
/// Loading the calibration object i based on the objects stored in the CCDB, which
/// is fully delegated to the framework CCDB support. Alternativelty, particularly for
/// testing purpose, local CCDB paths for files for the different calibration objects
/// can be provided. In this case the CCDB is bypassed. The function `static_load`
/// must always be called in the init function - it will take care of loading the
/// calibration objects to be taken from file. Furthermore in finalizeCCDB the workflow
/// must call finalizeCCDB from the CalibLoader in order to make load the calibration
/// objects requested from CCDB.
class CalibLoader
{
 public:
  /// \brief Constructor
  CalibLoader() = default;

  /// \brief Destructor
  ~CalibLoader() = default;

  /// \brief Access to current bad channel map
  /// \return Current bad channel map (nullptr if not loaded)
  BadChannelMap* getBadChannelMap() const { return mBadChannelMap; }

  /// \brief Access to current time calibration params
  /// \return Current time calibration params
  TimeCalibrationParams* getTimeCalibration() const { return mTimeCalibParams; }

  /// \brief Access to current gain calibration factors
  /// \return Current gain calibration factors
  GainCalibrationFactors* getGainCalibration() const { return mGainCalibParams; }

  /// \brief Check whether the bad channel map is handled
  /// \return True if the bad channel map is handled, false otherwise
  bool hasBadChannelMap() const { return mEnableBadChannelMap; }

  /// \brief Check whether the time calibration params are handled
  /// \return True if the time calibration params are handled, false otherwise
  bool hasTimeCalib() const { return mEnableTimeCalib; }

  /// \brief Check whether the gain calibration factors are handled
  /// \return True if the gain calibration factors are handled, false otherwise
  bool hasGainCalib() const { return mEnableGainCalib; }

  /// \brief Check whether the bad channel map has been updated
  /// \return True if the bad channel map has been updated, false otherwise
  bool hasUpdateBadChannelMap() const { return mUpdateStatus.test(OBJ_BADCHANNELMAP); }

  /// \brief Check whether the time calibration params have been updated
  /// \return True if the time calibration params have been updated, false otherwise
  bool hasUpdateTimeCalib() const { return mUpdateStatus.test(OBJ_TIMECALIB); }

  /// \brief Check whether the gain calibration params have been updated
  /// \return True if the gain calibration params have been updated, false  otherwise
  bool hasUpdateGainCalib() const { return mUpdateStatus.test(OBJ_GAINCALIB); }

  /// \brief Enable loading of the bad channel map
  /// \param doEnable If true the bad channel map is loaded (per default from CCDB)
  void enableBadChannelMap(bool doEnable) { mEnableBadChannelMap = doEnable; }

  /// \brief Enable loading of the time calibration params
  /// \param doEnable If true the time calibration params are loaded (per default from CCDB)
  void enableTimeCalib(bool doEnable) { mEnableTimeCalib = doEnable; }

  /// \brief Enable loading of the gain calibration factors
  /// \param doEnable If true the gain calibration factors are loaded (per default from CCDB)
  void enableGainCalib(bool doEnable) { mEnableGainCalib = doEnable; }

  /// \brief Mark bad channel map as updated
  void setUpdateBadChannelMap() { mUpdateStatus.set(OBJ_BADCHANNELMAP, true); }

  /// \brief Mark time calibration params as updated
  void setUpdateTimeCalib() { mUpdateStatus.set(OBJ_TIMECALIB, true); }

  /// \brief Mark gain calibration params as updated
  void setUpdateGainCalib() { mUpdateStatus.set(OBJ_GAINCALIB, true); }

  /// \brief Reset the update status (all objects marked as false)
  void resetUpdateStatus() { mUpdateStatus.reset(); }

  /// \brief Define input specs in workflow for calibration objects to be loaded from the CCDB
  /// \param ccdbInputs List of inputs where the CCDB input specs will be added to
  ///
  /// Defining only objects which are enabled and for which no local path is specified.
  void defineInputSpecs(std::vector<framework::InputSpec>& ccdbInputs);

  /// \brief Check for updates of the calibration objects in the processing context
  /// \param ctx ProcessingContext with InputSpecs
  ///
  /// Triggers finalizeCCDB in case of updates.
  void checkUpdates(o2::framework::ProcessingContext& ctx);

  /// \brief Callback for objects loaded from CCDB
  /// \param matcher Type of the CCDB object
  /// \param obj CCDB object loaded by the framework
  ///
  /// Only CCDB objects compatible with one of the types are handled. In case the object type
  /// is requested and not loaded from local source the object accepted locally and accessible
  /// via the corresponding getter function.
  bool finalizeCCDB(framework::ConcreteDataMatcher& matcher, void* obj);

 private:
  enum CalibObject_t {
    OBJ_BADCHANNELMAP,
    OBJ_TIMECALIB,
    OBJ_GAINCALIB
  };
  bool mEnableBadChannelMap;                                     ///< Switch for enabling / disabling loading of the bad channel map
  bool mEnableTimeCalib;                                         ///< Switch for enabling / disabling loading of the time calibration params
  bool mEnableGainCalib;                                         ///< Switch for enabling / disabling loading of the gain calibration params
  o2::emcal::BadChannelMap* mBadChannelMap = nullptr;            ///< Container of current bad channel map
  o2::emcal::TimeCalibrationParams* mTimeCalibParams = nullptr;  ///< Container of current time calibration object
  o2::emcal::GainCalibrationFactors* mGainCalibParams = nullptr; ///< Container of current gain calibration object
  std::bitset<16> mUpdateStatus;                                 ///< Object update status
};

} // namespace emcal

} // namespace o2
