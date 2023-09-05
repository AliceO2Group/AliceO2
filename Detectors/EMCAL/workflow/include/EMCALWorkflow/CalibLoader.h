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

#ifndef O2_EMCAL_CALIBLOADER
#define O2_EMCAL_CALIBLOADER

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
class FeeDCS;
class EMCALChannelScaleFactors;
class TimeCalibrationParams;
class GainCalibrationFactors;
class TempCalibrationParams;
class SimParam;
class RecoParam;

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

  /// \brief Access to current BCM Scale factors
  /// \return Current BCM scale factors (nullptr if not loaded)
  EMCALChannelScaleFactors* getBCMScaleFactors() const { return mBCMScaleFactors; }

  /// \brief Access to current FEE DCS params
  /// \return Current FEE DCS params (nullptr if not loaded)
  FeeDCS* getFEEDCS() const { return mFeeDCS; }

  /// \brief Access to current time calibration params
  /// \return Current time calibration params
  TimeCalibrationParams* getTimeCalibration() const { return mTimeCalibParams; }

  /// \brief Access to current gain calibration factors
  /// \return Current gain calibration factors
  GainCalibrationFactors* getGainCalibration() const { return mGainCalibParams; }

  /// \brief Access to current temperature calibration params
  /// \return Current temperature calibration factors
  TempCalibrationParams* getTemperatureCalibration() const { return mTempCalibParams; }

  /// \brief Check whether the bad channel map is handled
  /// \return True if the bad channel map is handled, false otherwise
  bool hasBadChannelMap() const { return mEnableStatus.test(OBJ_BADCHANNELMAP); }

  /// \brief Check whether the BCM scale factors are handled
  /// \return True if the BCM scale factors are handled, false otherwise
  bool hasBCMScaleFactors() const { return mEnableStatus.test(OBJ_BCMSCALEFACTORS); }

  /// \brief Check whether the FEE DCS params are handled
  /// \return True if the FEE DCS params are handled, false otherwise
  bool hasFEEDCS() const { return mEnableStatus.test(OBJ_FEEDCS); }

  /// \brief Check whether the time calibration params are handled
  /// \return True if the time calibration params are handled, false otherwise
  bool hasTimeCalib() const { return mEnableStatus.test(OBJ_TIMECALIB); }

  /// \brief Check whether the gain calibration factors are handled
  /// \return True if the gain calibration factors are handled, false otherwise
  bool hasGainCalib() const { return mEnableStatus.test(OBJ_GAINCALIB); }

  /// \brief Check whether the temperature calibration params are handled
  /// \return True if the temperature calibration params are handled, false otherwise
  bool hasTemperatureCalib() const { return mEnableStatus.test(OBJ_TEMPCALIB); }

  /// \brief Check whether the reconstruction params are handled
  /// \return True if the reconstruction params are handled, false otherwise
  bool hasRecoParams() const { return mEnableStatus.test(OBJ_RECOPARAM); }

  /// \brief Check whether the reconstruction params are handled
  /// \return True if the reconstruction params are handled, false otherwise
  bool hasSimParams() const { return mEnableStatus.test(OBJ_SIMPARAM); }

  /// \brief Check whether the bad channel map has been updated
  /// \return True if the bad channel map has been updated, false otherwise
  bool hasUpdateBadChannelMap() const { return mUpdateStatus.test(OBJ_BADCHANNELMAP); }

  /// \brief Check whether the BCM scale factors have been updated
  /// \return True if the bad channel map has been updated, false otherwise
  bool hasUpdateBCMScaleFactors() const { return mUpdateStatus.test(OBJ_BCMSCALEFACTORS); }

  /// \brief Check whether the FEE DCS params have been updated
  /// \return True if the FEE DCS params have been updated, false otherwise
  bool hasUpdateFEEDCS() const { return mUpdateStatus.test(OBJ_FEEDCS); }

  /// \brief Check whether the time calibration params have been updated
  /// \return True if the time calibration params have been updated, false otherwise
  bool hasUpdateTimeCalib() const { return mUpdateStatus.test(OBJ_TIMECALIB); }

  /// \brief Check whether the gain calibration params have been updated
  /// \return True if the gain calibration params have been updated, false otherwise
  bool hasUpdateGainCalib() const { return mUpdateStatus.test(OBJ_GAINCALIB); }

  /// \brief Check whether the temperature calibration params have been updated
  /// \return True if the temperature calibration params have been updated, false otherwise
  bool hasUpdateTemperatureCalib() const { return mUpdateStatus.test(OBJ_TEMPCALIB); }

  /// \brief Check whether the reconstruction params have been updated
  /// \return True if the reconstruction params have been updated, false otherwise
  bool hasUpdateRecoParam() const { return mUpdateStatus.test(OBJ_RECOPARAM); }

  /// \brief Check whether the simulation params have been updated
  /// \return True if the simulation params have been updated, false otherwise
  bool hasUpdateSimParam() const { return mUpdateStatus.test(OBJ_SIMPARAM); }

  /// \brief Enable loading of the bad channel map
  /// \param doEnable If true the bad channel map is loaded (per default from CCDB)
  void enableBadChannelMap(bool doEnable) { mEnableStatus.set(OBJ_BADCHANNELMAP, doEnable); }

  /// \brief Enable loading of the BCM scale factors
  /// \param doEnable If true the BCM scale factors are loaded (per default from CCDB)
  void enableBCMScaleFactors(bool doEnable) { mEnableStatus.set(OBJ_BCMSCALEFACTORS, doEnable); }

  /// \brief Enable loading of the FEE DCS params
  /// \param doEnable If true the FEE DCS params are loaded (per default from CCDB)
  void enableFEEDCS(bool doEnable) { mEnableStatus.set(OBJ_FEEDCS, doEnable); }

  /// \brief Enable loading of the time calibration params
  /// \param doEnable If true the time calibration params are loaded (per default from CCDB)
  void enableTimeCalib(bool doEnable) { mEnableStatus.set(OBJ_TIMECALIB, doEnable); }

  /// \brief Enable loading of the gain calibration factors
  /// \param doEnable If true the gain calibration factors are loaded (per default from CCDB)
  void enableGainCalib(bool doEnable) { mEnableStatus.set(OBJ_GAINCALIB, doEnable); }

  /// \brief Enable loading of the temperature calibration params
  /// \param doEnable If true the temperature calibration params are loaded (per default from CCDB)
  void enableTemperatureCalib(bool doEnable) { mEnableStatus.set(OBJ_TEMPCALIB, doEnable); }

  /// \brief Enable loading of the reconstruction params
  /// \param doEnable If true the reconstruction params are loaded (per default from CCDB)
  void enableRecoParams(bool doEnable) { mEnableStatus.set(OBJ_RECOPARAM, doEnable); }

  /// \brief Enable loading of the simulation params
  /// \param doEnable If true the simulation params are loaded (per default from CCDB)
  void enableSimParams(bool doEnable) { mEnableStatus.set(OBJ_SIMPARAM, doEnable); }

  /// \brief Mark bad channel map as updated
  void setUpdateBadChannelMap() { mUpdateStatus.set(OBJ_BADCHANNELMAP, true); }

  /// \brief Mark BCM scale factors as updated
  void setUpdateBCMScaleFactors() { mUpdateStatus.set(OBJ_BCMSCALEFACTORS, true); }

  /// \brief Mark FEE DCS params as updated
  void setUpdateFEEDCS() { mUpdateStatus.set(OBJ_FEEDCS, true); }

  /// \brief Mark time calibration params as updated
  void setUpdateTimeCalib() { mUpdateStatus.set(OBJ_TIMECALIB, true); }

  /// \brief Mark gain calibration params as updated
  void setUpdateGainCalib() { mUpdateStatus.set(OBJ_GAINCALIB, true); }

  /// \brief Mark temperature calibration params as updated
  void setUpdateTemperatureCalib() { mUpdateStatus.set(OBJ_TEMPCALIB, true); }

  /// \brief Mark reconstruction params as updated
  void setUpdateRecoParams() { mUpdateStatus.set(OBJ_RECOPARAM, true); }

  /// \brief Mark simulation params as updated
  void setUpdateSimParams() { mUpdateStatus.set(OBJ_SIMPARAM, true); }

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
    OBJ_BADCHANNELMAP,   ///< Bad channel map
    OBJ_TIMECALIB,       ///< Time calibration params
    OBJ_GAINCALIB,       ///< Energy calibration params
    OBJ_TEMPCALIB,       ///< Temperature calibration params
    OBJ_TIMSLEWINGCORR,  ///< Time slewing correction
    OBJ_BCMSCALEFACTORS, ///< BCM scale factors
    OBJ_FEEDCS,          ///< FEE DCS params
    OBJ_SIMPARAM,        ///< Simulation params
    OBJ_RECOPARAM,       ///< Reco params
    OBJ_CALIBPARAM       ///< Calibration params
  };
  o2::emcal::BadChannelMap* mBadChannelMap = nullptr;              ///< Container of current bad channel map
  o2::emcal::FeeDCS* mFeeDCS = nullptr;                            ///< Container of current FEE DCS params
  o2::emcal::TimeCalibrationParams* mTimeCalibParams = nullptr;    ///< Container of current time calibration object
  o2::emcal::GainCalibrationFactors* mGainCalibParams = nullptr;   ///< Container of current gain calibration object
  o2::emcal::TempCalibrationParams* mTempCalibParams = nullptr;    ///< Container of current temperature calibration object
  o2::emcal::EMCALChannelScaleFactors* mBCMScaleFactors = nullptr; ///< Container of current bad channel scale factors
  o2::emcal::RecoParam* mRecoParam;                                ///< Current reconstruction parameters
  o2::emcal::SimParam* mSimParam;                                  ///< Current simulation parameters
  std::bitset<16> mEnableStatus;                                   ///< Object enable status
  std::bitset<16> mUpdateStatus;                                   ///< Object update status
};

} // namespace emcal

} // namespace o2

#endif //  O2_EMCAL_CALIBLOADER
