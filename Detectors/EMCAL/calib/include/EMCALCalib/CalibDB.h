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
#include <exception>
#include <map>
#include <string>
#include "Rtypes.h"
#include "RStringView.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"

namespace o2
{

namespace emcal
{

class BadChannelMap;
class TempCalibrationParams;
class TempCalibParamSM;
class TimeCalibrationParams;
class TimeCalibParamL1Phase;
class GainCalibrationFactors;
class EMCALChannelScaleFactors;
class FeeDCS;
class ElmbData;
class Pedestal;

/// \class CalibDB
/// \brief Interface to calibration data from CCDB for EMCAL
/// \ingroup EMCALcalib
/// \author Markus Fasel <> Oak Ridge National Laboratory
/// \since July 12, 2019
///
/// Interface handling simple access to CCDB content for
/// EMCAL objects. The interface allows storing and
/// reading of the common EMCAL calibration objects
/// - Bad Channel Map
/// - Time calibration
/// - Gain calibration
/// - Temperature calibration
/// - Pedestals
/// Users only need to specify the CCDB server, the timestamp and
/// (optionally) additional meta data. Handling of the CCDB path
/// and type conversions is done internally - users deal directly
/// with the low level containers.
///
/// Attention: The read process of the CCDB objects might fail,
/// either because the query of the CCDB was not successfull
/// (wrong server / path / timestamp) or the object type is different
/// and internal conversion failed. In both cases dedicated exceptions
/// are thrown:
/// - ObjectNotFoundException in case of failure of the query
/// - TypeMismatchException in case object type doesn't match the expected type
/// Users must handle the exceptions.
class CalibDB
{

  using CcdbManager = o2::ccdb::BasicCCDBManager;

 public:
  /// \class ObjectNotFoundException
  /// \brief Handling errors due to objects not found in the CCDB
  ///
  /// Objects cannot be found due to
  /// - Not existing on server
  /// - Incorrect path
  /// - Wrong timestamp
  /// - Meta data not set
  class ObjectNotFoundException final : public std::exception
  {
   public:
    /// \brief Constructor with query parameters
    /// \param server URL of the CCDB server
    /// \param path CCDB path
    /// \param metadata Meta data used in the query
    /// \param timestamp Timestamp used in the query
    ObjectNotFoundException(const std::string_view server, const std::string_view path, const std::map<std::string, std::string>& metadata, ULong_t timestamp) : std::exception(),
                                                                                                                                                                 mServ(server),
                                                                                                                                                                 mPath(path),
                                                                                                                                                                 mMetaDat(metadata),
                                                                                                                                                                 mTimestamp(timestamp)
    {
      mMessage = "Not possible to access entry \"" + mPath + "\" on " + mServ + " for timestamp " + std::to_string(mTimestamp);
    }

    /// \brief destructor
    ~ObjectNotFoundException() noexcept final = default;

    /// \brief Creating error message with relevant query paramters
    /// \return error message
    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    /// \brief Accessor to meta data
    /// \return meta data
    const std::map<std::string, std::string> getMetaData() const { return mMetaDat; }

    /// \brief Accessor to URL of the CCDB server
    /// \return URL of the CCDB server
    const std::string& getServer() const { return mServ; }

    /// \brief Accessor to the CCDB path in the query
    /// \return CCDB path in the query
    const std::string& getPath() const { return mPath; }

    /// \brief Accessor to timestamp used in the query
    /// \return Timestamp used in query
    ULong_t getTimestamp() const { return mTimestamp; }

   private:
    const std::string mServ;                           ///< URL of the CCDB server
    const std::string mPath;                           ///< Query path
    std::string mMessage;                              ///< Resulting error message
    const std::map<std::string, std::string> mMetaDat; ///< <Meta data
    ULong_t mTimestamp;                                ///< Timestamp
  };

  /// \class TypeMismatchException
  /// \brief Class handling errors of wrong type of a query result
  ///
  /// The exepction is thrown in case the query for an object under
  /// a certain path and with a certain timestamp was valid, the object
  /// however has a different type than the expected one (something was
  /// screwed up when writing to the CCDB)
  class TypeMismatchException final : public std::exception
  {
   public:
    /// \brief Constructor
    /// \param obtained Type of the object obtained in the query
    /// \param expected Expected type of the object
    TypeMismatchException(const std::string_view obtained, const std::string_view expected) : std::exception(),
                                                                                              mTypeObtained(obtained),
                                                                                              mTypeExpected(expected),
                                                                                              mMessage()
    {
      mMessage = "Incorrect type, expected " + mTypeExpected + ", obtained " + mTypeObtained;
    }

    /// \brief Destructor
    ~TypeMismatchException() noexcept final = default;

    /// \brief Creating error message
    /// \return Error message with expected and obtained type
    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    /// \brief Accessor to expected type
    /// \return Name of the expected type
    const std::string& getExpectedType() const { return mTypeExpected; }

    /// \brief Accessor to the type of the object obtained from the CCDB
    /// \return Name of the obtained type of the object
    const std::string& getObtainedType() const { return mTypeObtained; }

   private:
    const std::string mTypeObtained; ///< Type of the object obtained from the CCDB
    const std::string mTypeExpected; ///< Expected type of the object
    std::string mMessage;            ///< Resulting error message
  };

  /// \brief Default constructor
  CalibDB() = default;

  /// \brief Constructor initializing also the server
  /// \param server Name of the CCDB server to be used in queries
  CalibDB(const std::string_view server);

  /// \brief Destructor
  ~CalibDB() = default;

  /// \brief Store bad channel map in the CCDB
  /// \param bcm Bad channel map to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeBadChannelMap(BadChannelMap* bcm, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find bad channel map in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  BadChannelMap* readBadChannelMap(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store time calibration coefficiencts in the CCDB
  /// \param tcp time calibration coefficiencts to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeTimeCalibParam(TimeCalibrationParams* tcp, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find time calibration coefficiencts in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  TimeCalibrationParams* readTimeCalibParam(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store L1 phase shifts in the CCDB
  /// \param tcp L1 phase shifts to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeTimeCalibParamL1Phase(TimeCalibParamL1Phase* tcp, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find L1 phase shifts in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  TimeCalibParamL1Phase* readTimeCalibParamL1Phase(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store temperature calibration coefficiencts in the CCDB
  /// \param tcp temperature calibration coefficiencts to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeTempCalibParam(TempCalibrationParams* tcp, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find temperature calibration coefficiencts in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  TempCalibrationParams* readTempCalibParam(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store temperature calibration coefficiencts per SM in the CCDB
  /// \param tcp temperature calibration coefficiencts per SM to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeTempCalibParamSM(TempCalibParamSM* tcp, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find temperature calibration coefficiencts per SM in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  TempCalibParamSM* readTempCalibParamSM(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store gain calibration factors in the CCDB
  /// \param gcf temperature calibration coefficiencts to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeGainCalibFactors(GainCalibrationFactors* gcf, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find gain calibration factors in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  GainCalibrationFactors* readGainCalibFactors(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Find scale factors used for bad channel calibration in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query (there is only one entry in the CCDB)
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  EMCALChannelScaleFactors* readChannelScaleFactors(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store FEE DCS data in the CCDB
  /// \param dcs FEE DCS data to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeFeeDCSData(FeeDCS* dcs, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find FEE DCS data in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  FeeDCS* readFeeDCSData(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store Temperature Sensor data in the CCDB
  /// \param dcs Temperature Sensor data to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storeTemperatureSensorData(ElmbData* dcs, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find Temperature Sensor data in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  ElmbData* readTemperatureSensorData(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Store pedestal data in the CCDB
  /// \param pedestals Pedestal data to be stored
  /// \param metadata Additional metadata that can be used in the query
  /// \param timestart Start of the time range of the validity of the object
  /// \param timeend End of the time range of the validity of the object
  void storePedestalData(Pedestal* pedestals, const std::map<std::string, std::string>& metadata, ULong_t timestart, ULong_t timeend);

  /// \brief Find pedestal data in the CCDB for given timestamp
  /// \param timestamp Timestamp used in query
  /// \param metadata Additional metadata to be used in the query
  /// \throw ObjectNotFoundException if object is not found for the given timestamp
  /// \throw TypeMismatchException if object is present but type is different (CCDB corrupted)
  Pedestal* readPedestalData(ULong_t timestamp, const std::map<std::string, std::string>& metadata);

  /// \brief Set new CCDB server URL
  /// \param server Name of the CCDB server to be used in queries
  ///
  /// Setting new CCDB server. Will require a re-init of the
  /// the CCDB handler the next time a store or read of any object
  /// is done.
  void setServer(const std::string_view server)
  {
    mCCDBServer = server;
    mInit = false;
  }

  /// \brief Get CDB path for the bad channel map
  /// \return Path of the bad channel map in the CCDB
  static const char* getCDBPathBadChannelMap() { return "EMC/Calib/BadChannelMap"; }

  /// \brief Get CDB path for the time calibration
  /// \return Path of the time calibration in the CCDB
  static const char* getCDBPathTimeCalibrationParams() { return "EMC/Calib/TimeCalibParams"; }

  /// \brief Get CDB path for the L1 Phase
  /// \return Path of the L1 phase in the CCDB
  static const char* getCDBPathL1Phase() { return "EMC/Calib/TimeCalibParamsL1Phase"; }

  /// \brief Get CDB path for the temperature calibration
  /// \return Path of the temperature calibration in the CCDB
  static const char* getCDBPathTemperatureCalibrationParams() { return "EMC/Calib/TempCalibParams"; }

  /// \brief Get CDB path for the SM-dependent temperature calibration
  /// \return Path of the SM-dependent temperature calibration in the CCDB
  static const char* getCDBPathTemperatureCalibrationParamsSM() { return "EMC/Calib/TempCalibParamsSM"; }

  /// \brief Get CDB path for the gain calibration
  /// \return Path of the gain calibration in the CCDB
  static const char* getCDBPathGainCalibrationParams() { return "EMC/Calib/GainCalibFactors"; }

  /// \brief Get CDB path for the FEE DCS settings
  /// \return Path of the FEE DCS settings in the CCDB
  static const char* getCDBPathFeeDCS() { return "EMC/Calib/FeeDCS"; }

  /// \brief Get CDB path for the Temperature Sensor data
  /// \return Path of the Temperature Sensor data in the CCDB
  static const char* getCDBPathTemperatureSensor() { return "EMC/Calib/Temperature"; }

  /// \brief Get CCDB path for the scale factors used in the bad channel calibration
  /// \return Path of the scale factors used in the bad channel calibration in the CCDB
  static const char* getCDBPathChannelScaleFactors() { return "EMC/Config/ChannelScaleFactors"; }

  /// \brief Get CCDB path for the pedestal data
  /// \return Path of the pedestal data
  static const char* getCDBPathChannelPedestals() { return "EMC/Calib/Pedestal"; }

 private:
  /// \brief Initialize CCDB server (when new object is created or the server URL changes)
  void
    init();

  ccdb::CcdbApi mCCDBManager;                       ///< Handler for queries of the CCDB content
  std::string mCCDBServer = "emcccdb-test.cern.ch"; ///< Name of the CCDB server
  Bool_t mInit = false;                             ///< Init status (needed for lazy evaluation of the CcdbApi init)

  ClassDefNV(CalibDB, 2);
};
} // namespace emcal

} // namespace o2
