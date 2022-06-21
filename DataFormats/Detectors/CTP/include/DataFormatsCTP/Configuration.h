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

/// \file Configuration.h
/// \brief definition of CTPConfiguration and related CTP structures
/// \author Roman Lietava

#ifndef _CTP_CONFIGURATION_H_
#define _CTP_CONFIGURATION_H_
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonConstants/LHCConstants.h"
#include "DataFormatsCTP/Scalers.h"
#include <string>
#include <vector>
#include <bitset>
#include <map>
#include <set>
namespace o2
{
namespace ctp
{
/// Database constants
const std::string CCDBPathCTPConfig = "CTP/Config/Config";
///
/// CTP Config items
struct BCMask {
  BCMask() = default;
  std::string name = "";
  std::string mask = "";
  std::bitset<o2::constants::lhc::LHCMaxBunches> BCmask;
  void printStream(std::ostream& stream) const;
  ClassDefNV(BCMask, 1);
};
struct CTPGenerator {
  static const std::set<std::string> Generators;
  std::string name = "";
  std::string frequency = "";
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPGenerator, 1);
};
struct CTPInput {
  CTPInput() = default;
  std::string name = "";
  std::string level = "";
  std::uint64_t inputMask = 0;
  o2::detectors::DetID::ID detID;
  std::string getInputDetName() const { return o2::detectors::DetID::getName(detID); }
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPInput, 2);
};
struct CTPDescriptor {
  CTPDescriptor() = default;
  std::string name = "";
  std::vector<CTPInput*> inputs;
  std::uint64_t getInputsMask() const;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPDescriptor, 2)
};
/// The main part is Local Trigger Generator (LTG)
struct CTPDetector {
  CTPDetector() = default;
  o2::detectors::DetID::ID detID;
  const char* getName() const { return o2::detectors::DetID::getName(detID); }
  uint32_t HBaccepted; /// Number of HB frames in TF to be accepted
  std::string mode = "";
  uint32_t ferst = 0;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPDetector, 1)
};
struct CTPCluster {
  CTPCluster() = default;
  std::string name = "";
  uint32_t hwMask = 0;
  o2::detectors::DetID::mask_t maskCluster;
  std::string getClusterDetNames() const { return o2::detectors::DetID::getNames(maskCluster, ' '); }
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPCluster, 3)
};
struct CTPClass {
  CTPClass() = default;
  std::string name = "";
  std::uint64_t classMask = 0;
  CTPDescriptor const* descriptor = nullptr;
  CTPCluster const* cluster = nullptr;
  int clusterIndex = 0;
  int descriptorIndex = 0;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPClass, 3);
};
class CTPConfiguration
{
 public:
  CTPConfiguration() = default;
  bool isDetector(const o2::detectors::DetID& det);
  void capitaliseString(std::string& str);
  enum ConfigPart { RUN,
                    MASKS,
                    GENS,
                    INPUT,
                    LTG,
                    LTGitems,
                    CLUSTER,
                    CLASS,
                    UNKNOWN };
  int loadConfigurationRun3(const std::string& ctpconfiguartion);
  void printStream(std::ostream& stream) const;
  std::vector<CTPInput>& getCTPInputs() { return mInputs; }
  std::vector<CTPClass>& getCTPClasses() { return mCTPClasses; }
  uint64_t getInputMask(const std::string& name);
  bool isMaskInInputs(const uint64_t& mask) const;
  bool isBCMaskInConfig(const std::string maskname) const;
  CTPInput* isInputInConfig(const std::string inpname);
  uint64_t getDecrtiptorInputsMask(const std::string& name) const;
  std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> getDet2InputMap();
  uint64_t getTriggerClassMask() const;
  std::vector<int> getTriggerClassList() const;
  uint32_t getRunNumber() { return mRunNumber; };

 private:
  std::string mConfigString = "";
  uint32_t mRunNumber = 0;
  std::string mName = "";
  std::string mVersion = "0";
  std::vector<BCMask> mBCMasks;
  std::vector<CTPGenerator> mGenerators;
  std::vector<CTPInput> mInputs;
  std::vector<CTPDescriptor> mDescriptors;
  std::vector<CTPDetector> mDetectors;
  std::vector<CTPCluster> mClusters;
  std::vector<CTPClass> mCTPClasses;
  int processConfigurationLineRun3(std::string& line, int& level);
  ClassDefNV(CTPConfiguration, 4);
};
// Run Manager
struct CTPActiveRun {
  CTPActiveRun() = default;
  long timeStart;
  long timeStop;
  CTPConfiguration cfg;
  CTPRunScalers scalers;
};
class CTPRunManager
{
 public:
  CTPRunManager() = default;
  void init();
  int startRun(const std::string& cfg);
  int stopRun(uint32_t irun);
  int addScalers(uint32_t irun, std::time_t time);
  int processMessage(std::string& topic, const std::string& message);
  void printActiveRuns() const;
  int saveRunScalersToCCDB(int i);
  int saveRunConfigToCCDB(CTPConfiguration* cfg, long timeStart);
  CTPConfiguration getConfigFromCCDB(long timestamp, std::string run);
  CTPRunScalers getScalersFromCCDB(long timestamp, std::string);
  int loadScalerNames();
  void setCcdbHost(std::string host) { mCcdbHost = host; };
  // void setCCDBPathConfig(std::string path) { mCCDBPathCTPConfig = path;};
  void setCCDBPathScalers(std::string path) { mCCDBPathCTPScalers = path; };
  void printCounters();

 private:
  /// Database constants
  // std::string mCcdbHost = "http://ccdb-test.cern.ch:8080";
  std::string mCcdbHost = "http://o2-ccdb.internal:8080";
  std::string mCCDBPathCTPScalers = "CTP/Calib/Scalers";
  std::array<CTPActiveRun*, NRUNS> mActiveRuns;
  std::array<std::uint32_t, NRUNS> mActiveRunNumbers;
  std::array<uint32_t, CTPRunScalers::NCOUNTERS> mCounters;
  std::map<std::string, uint32_t> mScalerName2Position;
  CTPActiveRun* mRunInStart = nullptr;
  int mEOX = 0; // redundancy check
  int mCtpcfg = 0;
  ClassDefNV(CTPRunManager, 3);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_CONFIGURATION_H_
