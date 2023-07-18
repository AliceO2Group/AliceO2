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

/// \file CDBInterface.h
/// \brief Simple interface to the CDB manager
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_CDBInterface_H_
#define AliceO2_TPC_CDBInterface_H_

#include <memory>
#include <unordered_map>
#include <string_view>

#include "Framework/Logger.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "TPCBase/CalDet.h"
#include "DataFormatsTPC/LtrCalibData.h"
#include "CommonUtils/NameConf.h"

namespace o2::tpc
{
// forward declarations
class ParameterDetector;
class ParameterElectronics;
class ParameterGas;
class ParameterGEM;

/// Calibration and parameter types for CCDB
enum class CDBType {
  CalPedestal,         ///< Pedestal calibration
  CalNoise,            ///< Noise calibration
  CalPedestalNoise,    ///< Pedestal and Noise calibration
  CalPulser,           ///< Pulser calibration
  CalCE,               ///< Laser CE calibration
  CalPadGainFull,      ///< Full pad gain calibration
  CalPadGainResidual,  ///< ResidualpPad gain calibration (e.g. from tracks)
  CalLaserTracks,      ///< Laser track calibration data
  CalVDriftTgl,        ///< ITS-TPC difTgl vdrift calibration
  CalTimeGain,         ///< Gain variation over time
  CalGas,              ///< DCS gas measurements
  CalTemperature,      ///< DCS temperature measurements
  CalHV,               ///< DCS HV measurements
  CalTopologyGain,     ///< Q cluster topology correction
                       ///
  ConfigFEEPad,        ///< FEE pad-by-pad configuration map
  ConfigFEE,           ///< FEE configuration map for each tag
  ConfigRunInfo,       ///< FEE run information (run -> tag)
                       ///
  ParDetector,         ///< Parameter for Detector
  ParElectronics,      ///< Parameter for Electronics
  ParGas,              ///< Parameter for Gas
  ParGEM,              ///< Parameter for GEM
                       ///
  CalIDC0A,            ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  CalIDC0C,            ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  CalIDC1A,            ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  CalIDC1C,            ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  CalIDCDeltaA,        ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  CalIDCDeltaC,        ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  CalIDCFourierA,      ///< Fourier coefficients of CalIDC1
  CalIDCFourierC,      ///< Fourier coefficients of CalIDC1
  CalIDCPadStatusMapA, ///< Status map of the pads (dead etc. obatined from CalIDC0)
  CalIDCPadStatusMapC, ///< Status map of the pads (dead etc. obatined from CalIDC0)
  CalIDCGroupingParA,  ///< Parameters which were used for the averaging of the CalIDCDelta
  CalIDCGroupingParC,  ///< Parameters which were used for the averaging of the CalIDCDelta
                       ///
  CalSAC0,             ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  CalSAC1,             ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  CalSACDelta,         ///< \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  CalSACFourier,       ///< Fourier coefficients of CalSAC1
                       ///
  CalITPC0,            ///< 2D average TPC clusters for longer time interval
  CalITPC1,            ///< 1D integrated TPC clusters
                       ///
  CalCorrMap,          ///< Cluster correction map
  CalCorrMapRef,       ///< Cluster correction reference map (static distortions)
};

/// Upload intervention type
enum class CDBIntervention {
  Manual,    ///< Upload from manual intervention
  Automatic, ///< Automatic upload
};

/// Storage name in CCDB for each calibration and parameter type
const std::unordered_map<CDBType, const std::string> CDBTypeMap{
  {CDBType::CalPedestal, "TPC/Calib/Pedestal"},
  {CDBType::CalNoise, "TPC/Calib/Noise"},
  {CDBType::CalPedestalNoise, "TPC/Calib/PedestalNoise"},
  {CDBType::CalPulser, "TPC/Calib/Pulser"},
  {CDBType::CalCE, "TPC/Calib/CE"},
  {CDBType::CalPadGainFull, "TPC/Calib/PadGainFull"},
  {CDBType::CalPadGainResidual, "TPC/Calib/PadGainResidual"},
  {CDBType::CalLaserTracks, "TPC/Calib/LaserTracks"},
  {CDBType::CalTimeGain, "TPC/Calib/TimeGain"},
  {CDBType::CalGas, "TPC/Calib/Gas"},
  {CDBType::CalTemperature, "TPC/Calib/Temperature"},
  {CDBType::CalHV, "TPC/Calib/HV"},
  {CDBType::CalTopologyGain, "TPC/Calib/TopologyGainPiecewise"},
  {CDBType::CalVDriftTgl, "TPC/Calib/VDriftTgl"},
  //
  {CDBType::ConfigFEEPad, "TPC/Config/FEEPad"},
  {CDBType::ConfigFEE, "TPC/Config/FEE"},
  {CDBType::ConfigRunInfo, "TPC/Config/RunInfo"},
  //
  {CDBType::ParDetector, "TPC/Parameter/Detector"},
  {CDBType::ParElectronics, "TPC/Parameter/Electronics"},
  {CDBType::ParGas, "TPC/Parameter/Gas"},
  {CDBType::ParGEM, "TPC/Parameter/GEM"},
  // IDCs
  {CDBType::CalIDC0A, "TPC/Calib/IDC_0_A"},
  {CDBType::CalIDC0C, "TPC/Calib/IDC_0_C"},
  {CDBType::CalIDC1A, "TPC/Calib/IDC_1_A"},
  {CDBType::CalIDC1C, "TPC/Calib/IDC_1_C"},
  {CDBType::CalIDCDeltaA, "TPC/Calib/IDC_DELTA_A"},
  {CDBType::CalIDCDeltaC, "TPC/Calib/IDC_DELTA_C"},
  {CDBType::CalIDCFourierA, "TPC/Calib/IDC_FOURIER_A"},
  {CDBType::CalIDCFourierC, "TPC/Calib/IDC_FOURIER_C"},
  {CDBType::CalIDCPadStatusMapA, "TPC/Calib/IDC_PadStatusMap_A"},
  {CDBType::CalIDCPadStatusMapC, "TPC/Calib/IDC_PadStatusMap_C"},
  {CDBType::CalIDCGroupingParA, "TPC/Calib/IDC_GROUPINGPAR_A"},
  {CDBType::CalIDCGroupingParC, "TPC/Calib/IDC_GROUPINGPAR_C"},
  // SACs
  {CDBType::CalSAC0, "TPC/Calib/SAC_0"},
  {CDBType::CalSAC1, "TPC/Calib/SAC_1"},
  {CDBType::CalSACDelta, "TPC/Calib/SAC_DELTA"},
  {CDBType::CalSACFourier, "TPC/Calib/SAC_FOURIER"},
  // ITPCCs
  {CDBType::CalITPC0, "TPC/Calib/ITPCC_0"},
  {CDBType::CalITPC1, "TPC/Calib/ITPCC_1"},
  // correction maps
  {CDBType::CalCorrMap, "TPC/Calib/CorrectionMap"},
  {CDBType::CalCorrMapRef, "TPC/Calib/CorrectionMapRef"},
};

/// Poor enum reflection ...
const std::unordered_map<CDBIntervention, std::string> CDBInterventionMap{
  {CDBIntervention::Manual, "Manual"},
  {CDBIntervention::Automatic, "Automatic"},
};

/// \class CDBInterface
/// The class provides a simple interface to the CDB for the TPC specific
/// objects. It will not take ownership of the objects, but will leave this
/// to the CDB itself.
/// This class is used in the simulation and reconstruction as a singleton.
/// For local tests it offers the possibility to return default values.
/// To use this one needs to call
/// <pre>CDBInterface::instance().setUseDefaults();</pre>
/// at some point.
/// It also allows to specifically load pedestals and noise from file using the
/// <pre>loadNoiseAndPedestalFromFile(...)</pre> function
class CDBInterface
{
 public:
  using CalPadMapType = std::unordered_map<std::string, CalPad>;

  CDBInterface(const CDBInterface&) = delete;

  /// Create instance of singleton
  /// \return singleton instance
  static CDBInterface& instance()
  {
    static CDBInterface interface;
    return interface;
  }

  /// Return the pedestal object
  ///
  /// The function checks if the object is already loaded and returns it
  /// otherwise the object will be loaded first depending on the configuration
  /// \return pedestal object
  const CalPad& getPedestals();

  /// Return the pedestal object with 10+2bit precision as used in the CRU
  ///
  /// \return pedestal object as used in the CRU
  const CalPad& getPedestalsCRU();

  /// Return the noise object
  ///
  /// The function checks if the object is already loaded and returns it
  /// otherwise the object will be loaded first depending on the configuration
  /// \return noise object
  const CalPad& getNoise();

  /// Return the zero suppression threshold map
  const CalPad& getZeroSuppressionThreshold();

  /// Return the gain map object
  ///
  /// The function checks if the object is already loaded and returns it
  /// otherwise the object will be loaded first depending on the configuration
  /// \return gain map object
  const CalPad& getGainMap();

  /// Return the ion tail coupling fraction
  ///
  /// \return ion tail fraction
  const CalPad& getITFraction();

  /// Return the ion tail exponential decay parameter
  ///
  /// \return ion tail exponential decay parameter
  const CalPad& getITExpLambda();

  /// Return the common mode per pad values
  ///
  /// \return common mode per pad values
  const CalPad& getCMkValues();

  /// Return the Detector parameters
  ///
  /// The function checks if the object is already loaded and returns it
  /// otherwise the object will be loaded first depending on the configuration
  /// \return Detector parameters
  const ParameterDetector& getParameterDetector();

  /// Return the Electronics parameters
  ///
  /// The function checks if the object is already loaded and returns it
  /// otherwise the object will be loaded first depending on the configuration
  /// \return Electronics parameters
  const ParameterElectronics& getParameterElectronics();

  /// Return the Gas parameters
  ///
  /// The function checks if the object is already loaded and returns it
  /// otherwise the object will be loaded first depending on the configuration
  /// \return Gas parameters
  const ParameterGas& getParameterGas();

  /// Return the GEM parameters
  ///
  /// The function checks if the object is already loaded and returns it
  /// otherwise the object will be loaded first depending on the configuration
  /// \return GEM parameters
  const ParameterGEM& getParameterGEM();

  /// Return a CalPad object form the CCDB
  /// Deprecated
  const CalPad& getCalPad(const std::string_view path);

  /// Return any templated object
  ///
  /// The function returns the object stored at the given path, timestamp and metaData in the CCDB
  /// \return object
  template <typename T>
  T& getSpecificObjectFromCDB(const std::string_view path, long timestamp = -1, const std::map<std::string, std::string>& metaData = std::map<std::string, std::string>());

  /// read an object from CCDB
  template <typename T>
  T& getObjectFromCDB(std::string_view path);

  /// Set noise and pedestal object from file
  ///
  /// This assumes that the objects are stored under the name
  /// 'Pedestals' and 'Noise', respectively
  ///
  /// \param fileName name of the file containing pedestals and noise
  void setPedestalsAndNoiseFromFile(const std::string_view fileName) { mPedestalNoiseFileName = fileName; }

  /// Set gain map from file
  ///
  /// This assumes that the objects is stored under the name 'Gain'
  ///
  /// \param fileName name of the file containing gain map
  void setGainMapFromFile(const std::string_view fileName) { mGainMapFileName = fileName; }

  /// Set zero suppression thresholds from file (will load all FEE params)
  ///
  /// This assumes that the objects is stored under the name 'ThresholdMap'
  ///
  /// \param fileName name of the file containing the threshold map
  void setThresholdMapFromFile(const std::string_view fileName) { mFEEParamFileName = fileName; }

  /// Set FEE parameters from file
  ///
  /// This assumes that the objects are stored under the name 'fraction,expLambda'
  ///
  /// \param fileName name of the file containing the ion tail parameters
  void setFEEParamsFromFile(const std::string_view fileName) { mFEEParamFileName = fileName; }

  /// Force using default values instead of reading the CCDB
  ///
  /// \param default switch if to use default values
  void setUseDefaults(bool defaults = true) { mUseDefaults = defaults; }

  /// return defaults usage
  bool getUseDefaults() const { return mUseDefaults; }

  /// set CDB time stamp for object retrieval
  void setTimeStamp(long time)
  {
    auto& cdb = o2::ccdb::BasicCCDBManager::instance();
    cdb.setTimestamp(time);
  }

  /// set CCDB URL
  void setURL(const std::string_view url)
  {
    auto& cdb = o2::ccdb::BasicCCDBManager::instance();
    cdb.setURL(url.data());
  }

  /// set the Zero suppression threshold in sigma of noise in case
  /// the default object is created and not loaded from file or ccdb
  void setDefaultZSsigma(float zs)
  {
    mDefaultZSsigma = zs;
  }

  /// Reset the local calibration
  void resetLocalCalibration()
  {
    mPedestals.reset();
    mNoise.reset();
    mZeroSuppression.reset();
    mGainMap.reset();
  }

 private:
  CDBInterface() = default;

  // ===| Pad calibrations |====================================================
  std::unique_ptr<CalPad> mPedestals;       ///< Pedestal object
  std::unique_ptr<CalPad> mPedestalsCRU;    ///< Pedestal object with 10+2bit precision as used in CRU
  std::unique_ptr<CalPad> mNoise;           ///< Noise object
  std::unique_ptr<CalPad> mZeroSuppression; ///< Zero suppression object
  std::unique_ptr<CalPad> mGainMap;         ///< Gain map object
  std::unique_ptr<CalPad> mITFraction;      ///< Ion Tail fraction
  std::unique_ptr<CalPad> mITExpLambda;     ///< Ion Tail exp(-lambda)
  std::unique_ptr<CalPad> mCMkValues;       ///< Ion Tail exp(-lambda)

  // ===| switches and parameters |=============================================
  bool mUseDefaults = false;   ///< use defaults instead of CCDB
  float mDefaultZSsigma = 3.f; ///< sigma to use in case the default zero suppression is created

  std::string mPedestalNoiseFileName; ///< optional file name for pedestal and noise data
  std::string mGainMapFileName;       ///< optional file name for the gain map
  std::string mFEEParamFileName;      ///< optional file name for the FEE parameters (ion tail, common mode, threshold, pedestals)

  // ===========================================================================
  // ===| functions |===========================================================
  //
  void loadNoiseAndPedestalFromFile(); ///< load noise and pedestal values from mPedestalNoiseFileName
  void loadGainMapFromFile();          ///< load gain map from mGainmapFileName
  void loadFEEParamsFromFile();        ///< load ion tail paramters

  void createDefaultPedestals();       ///< creation of default pedestals if requested
  void createDefaultNoise();           ///< creation of default noise if requested
  void createDefaultZeroSuppression(); ///< creation of default noise if requested
  void createDefaultGainMap();         ///< creation of default gain map if requested
  void createDefaultIonTailParams();   ///< creation of default ion tail parameters
  void createDefaultCMParams();        ///< creation of default common mode parameters
};

/// Get an object from the CCDB.
/// @tparam T
/// @param path
/// @return The object from the CCDB, ownership is transferred to the caller.
/// @todo Consider removing in favour of calling directly the manager::get method.
template <typename T>
inline T& CDBInterface::getObjectFromCDB(std::string_view path)
{
  static auto& cdb = o2::ccdb::BasicCCDBManager::instance();
  auto* object = cdb.get<T>(path.data());
  if (!object) {
    LOGP(fatal, "Could not get {} from cdb", path);
  }
  return *object;
}

/// Get a CalPad object stored in templated formats from the CCDB.
/// @tparam T
/// @param path
/// @param timestamp
/// @param metaData
/// @return The object from the CCDB, ownership is transferred to the caller.
/// @todo Consider removing in favour of calling directly the manager::get method.
template <typename T>
inline T& CDBInterface::getSpecificObjectFromCDB(std::string_view path, long timestamp, const std::map<std::string, std::string>& metaData)
{
  static auto& cdb = o2::ccdb::BasicCCDBManager::instance();
  auto* object = cdb.getSpecific<T>(path.data(), timestamp, metaData);
  return *object;
}

template CalPad& CDBInterface::getSpecificObjectFromCDB(const std::string_view path, long timestamp, const std::map<std::string, std::string>& metaData);
template std::vector<CalPad>& CDBInterface::getSpecificObjectFromCDB(const std::string_view path, long timestamp, const std::map<std::string, std::string>& metaData);
template CDBInterface::CalPadMapType& CDBInterface::getSpecificObjectFromCDB(const std::string_view path, long timestamp, const std::map<std::string, std::string>& metaData);
template LtrCalibData& CDBInterface::getSpecificObjectFromCDB(const std::string_view path, long timestamp, const std::map<std::string, std::string>& metaData);

/// \class CDBStorage
/// Simple interface to store TPC CCDB types. Also provide interface functions to upload data from
/// a file.
///
///
class CDBStorage
{
 public:
  using MetaData_t = std::map<std::string, std::string>;
  void setURL(std::string_view url)
  {
    mCCDB.init(url.data());
  }

  void clearMetaData()
  {
    mMetaData.clear();
  }

  void setResponsible(std::string_view responsible)
  {
    mMetaData["Responsible"] = responsible;
  }

  void setReason(std::string_view reason)
  {
    mMetaData["Reason"] = reason;
  }

  void setIntervention(CDBIntervention const intervention)
  {
    mMetaData["Intervention"] = CDBInterventionMap.at(intervention);
  }

  void setJIRA(std::string_view jira)
  {
    mMetaData["JIRA"] = jira;
  }

  void setComment(std::string_view comment)
  {
    mMetaData["Comment"] = comment;
  }

  void setRunNumber(int run)
  {
    mMetaData[o2::base::NameConf::CCDBRunTag.data()] = std::to_string(run);
  }

  const auto& getMetaData() const { return mMetaData; }

  void setSimulate(bool sim = true) { mSimulate = sim; }

  bool getSimulate() const { return mSimulate; }

  template <typename T>
  void storeObject(T* obj, CDBType const type, MetaData_t const& metadata, long start, long end)
  {
    if (checkMetaData(metadata)) {
      if (!mSimulate) {
        mCCDB.storeAsTFileAny(obj, CDBTypeMap.at(type), metadata, start, end);
      }
      printObjectSummary(typeid(obj).name(), type, metadata, start, end);
    } else {
      LOGP(error, "Meta data not set properly, object will not be stored");
    }
  }

  template <typename T>
  void storeObject(T* obj, CDBType const type, long start, long end)
  {
    storeObject(obj, type, mMetaData, start, end);
  }

  void uploadNoiseAndPedestal(std::string_view fileName, long first = -1, long last = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
  void uploadGainMap(std::string_view fileName, bool isFull = true, long first = -1, long last = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
  void uploadPulserOrCEData(CDBType type, std::string_view fileName, long first = -1, long last = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
  void uploadFEEConfigPad(std::string_view fileName, long first = -1, long last = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
  void uploadTimeGain(std::string_view fileName, long first = -1, long last = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);

 private:
  bool checkMetaData(MetaData_t metaData) const;
  bool checkMetaData() const { return checkMetaData(mMetaData); }
  void printObjectSummary(std::string_view name, CDBType const type, MetaData_t const& metadata, long start, long end) const;

  o2::ccdb::CcdbApi mCCDB;
  MetaData_t mMetaData;
  bool mSimulate = false;
};

} // namespace o2::tpc

#endif
