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

#include <string>
#include <string_view>
#include <vector>
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
  BadChannelMap* getBadChannelMap() const { return mBadChannelMap.getObject(); }

  /// \brief Access to current time calibration params
  /// \return Current time calibration params
  TimeCalibrationParams* getTimeCalibration() const { return mTimeCalibParams.getObject(); }

  /// \brief Access to current gain calibration factors
  /// \return Current gain calibration factors
  GainCalibrationFactors* getGainCalibration() const { return mGainCalibParams.getObject(); }

  /// \brief Check whether the bad channel map is handled
  /// \return True if the bad channel map is handled, false otherwise
  bool hasBadChannelMap() const { return mEnableBadChannelMap; }

  /// \brief Check whether the time calibration params are handled
  /// \return True if the time calibration params are handled, false otherwise
  bool hasTimeCalib() const { return mEnableTimeCalib; }

  /// \brief Check whether the gain calibration factors are handled
  /// \return True if the gain calibration factors are handled, false otherwise
  bool hasGainCalib() const { return mEnableGainCalib; }

  /// \brief Check whether the bad channel map is loaded from local file
  /// \return True if the bad channel map is loaded from a local file
  bool hasLocalBadChannelMap() const { return mEnableBadChannelMap && mPathBadChannelMap.length(); }

  /// \brief Check whether the time calibration params are loaded from local file
  /// \return True if the time calibration params are loaded from a local file
  bool hasLocalTimeCalib() const { return mEnableTimeCalib && mPathTimeCalib.length(); }

  /// \brief Check whether the gain calibration factors are loaded from local file
  /// \return True if the gain calibration factors are loaded from a local file
  bool hasLocalGainCalib() const { return mEnableGainCalib && mPathGainCalib.length(); }

  /// \brief Enable loading of the bad channel map
  /// \param doEnable If true the bad channel map is loaded (per default from CCDB)
  void enableBadChannelMap(bool doEnable) { mEnableBadChannelMap = doEnable; }

  /// \brief Enable loading of the time calibration params
  /// \param doEnable If true the time calibration params are loaded (per default from CCDB)
  void enableTimeCalib(bool doEnable) { mEnableTimeCalib = doEnable; }

  /// \brief Enable loading of the gain calibration factors
  /// \param doEnable If true the gain calibration factors are loaded (per default from CCDB)
  void enableGainCalib(bool doEnable) { mEnableGainCalib = doEnable; }

  /// \brief Define path of local fine for bad channel map
  /// \param path Path of local file with bad channel map
  ///
  /// The bad channel map will be taken from the file instead of the CCDB.
  /// The file is expected to contain a valid o2::emcal::BadChannelMap with
  /// the name "ccdb_object"
  void setLoadBadChannelMapFromFile(const std::string_view path)
  {
    mPathBadChannelMap = path;
    enableBadChannelMap(true);
  }

  /// \brief Define path of local fine for time calibration params
  /// \param path Path of local file with time calibration params
  ///
  /// The time calibration params will be taken from the file instead
  /// of the CCDB. The file is expected to contain a valid
  /// o2::emcal::TimeCalibrationParams with the name "ccdb_object"
  void setLoadTimeCalibFromFile(const std::string_view path)
  {
    mPathTimeCalib = path;
    enableBadChannelMap(true);
  }

  /// \brief Define path of local fine for gain calibration factors
  /// \param path Path of local file with gain calibration factors
  ///
  /// The gain calibration factors will be taken from the file instead
  /// of the CCDB. The file is expected to contain a valid
  /// o2::emcal::GainCalibrationFactors with the name "ccdb_object"
  void setLoadGainCalibFromFile(const std::string_view path)
  {
    mPathGainCalib = path;
    enableGainCalib(true);
  }

  /// \brief Define input specs in workflow for calibration objects to be loaded from the CCDB
  /// \param ccdbInputs List of inputs where the CCDB input specs will be added to
  ///
  /// Defining only objects which are enabled and for which no local path is specified.
  void defineInputSpecs(std::vector<framework::InputSpec>& ccdbInputs);

  /// \brief Callback for objects loaded from CCDB
  /// \param matcher Type of the CCDB object
  /// \param obj CCDB object loaded by the framework
  ///
  /// Only CCDB objects compatible with one of the types are handled. In case the object type
  /// is requested and not loaded from local source the object accepted locally and accessible
  /// via the corresponding getter function.
  bool finalizeCCDB(framework::ConcreteDataMatcher& matcher, void* obj);

  /// \brief Load all calibration objects for which a local path is defined
  void static_load();

 private:
  /// \class ManagedObject
  /// \brief Internal helper treating owned and non-owned object with the same member
  /// \tparam T object type to be handled
  ///
  /// In case the object is owned the object will be deleted in case the helper goes
  /// out-of-scope. Otherwise ownership will be assumed with someone else and the object
  /// will not be deleted in the destructor.
  template <typename T>
  class ManagedObject
  {
   public:
    /// \brief Dummy constructor
    ManagedObject() = default;

    /// \brief Constructor, defining the object as managed or not
    /// \param object Object to be handled
    /// \param managed If true the object will be automatically deleted if the helper goes out-of-scope
    ManagedObject(T* object, bool managed) : mObject(object), mManaged(managed) {}

    /// \brief Destructor, deletes only managed objects
    ~ManagedObject()
    {
      if (mManaged) {
        delete mObject;
      }
    }

    /// \brief Access to the object
    /// \return Object handled by the helper
    T* getObject() const { return mObject; }

   private:
    T* mObject = nullptr;  ///< Object to be handled
    bool mManaged = false; ///< Management status
  };

  bool mEnableBadChannelMap;                                         ///< Switch for enabling / disabling loading of the bad channel map
  bool mEnableTimeCalib;                                             ///< Switch for enabling / disabling loading of the time calibration params
  bool mEnableGainCalib;                                             ///< Switch for enabling / disabling loading of the gain calibration params
  std::string mPathBadChannelMap;                                    ///< Path of local file with bad channel map (optional)
  std::string mPathTimeCalib;                                        ///< Path of local file with time calibration params (optional)
  std::string mPathGainCalib;                                        ///< Path of local file with gain calibration params (optional)
  ManagedObject<o2::emcal::BadChannelMap> mBadChannelMap;            ///< Container of current bad channel map
  ManagedObject<o2::emcal::TimeCalibrationParams> mTimeCalibParams;  ///< Container of current time calibration object
  ManagedObject<o2::emcal::GainCalibrationFactors> mGainCalibParams; ///< Container of current gain calibration object
};

} // namespace emcal

} // namespace o2
