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
#include <iostream>
namespace o2
{
namespace ctp
{
/// Database constants
const std::string CCDBPathCTPConfig = "CTP/Config/Config";
///
/// CTP Config items
///
// Bunch Crossing (BC) mask
struct BCMask {
  BCMask() = default;
  std::string name = "";
  std::string mask = "";
  std::bitset<o2::constants::lhc::LHCMaxBunches> BCmask;
  int setBCmask(std::vector<std::string>& tokens);
  void printStream(std::ostream& stream) const;
  ClassDefNV(BCMask, 1);
};
/// CTP internal generator: 4 for L0 and 4 for LM levels
struct CTPGenerator {
  static const std::set<std::string> Generators;
  std::string name = "";
  std::string frequency = "";
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPGenerator, 1);
};
/// CTP inputs
/// Default input config is in CTPConfiguration
struct CTPInput {
  const static std::map<std::string, std::string> run2DetToRun3Det;
  CTPInput() = default;
  CTPInput(std::string& name, std::string& det, uint32_t index);
  CTPInput(const char* name, const char* det, uint32_t index);
  std::string name = "";
  std::string level = "";
  std::uint64_t inputMask = 0;
  o2::detectors::DetID::ID detID = 16; // CTP
  bool neg = 1;
  int getIndex() const { return ((inputMask > 0) ? 1 + log2(inputMask) : 0xff); }
  std::string getInputDetName() const { return o2::detectors::DetID::getName(detID); }
  void setRun3DetName(std::string& run2Name);
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPInput, 3);
};
/// Descriptor = Generator or List of [negated] inputs
struct CTPDescriptor {
  CTPDescriptor() = default;
  std::string name = "";
  std::vector<CTPInput const*> inputs;
  std::uint64_t getInputsMask() const;
  // void createInputsFromName();
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPDescriptor, 2)
};
/// The main part is Local Trigger Generator (LTG)
struct CTPDetector {
  CTPDetector() = default;
  o2::detectors::DetID::ID detID;
  o2::detectors::DetID::mask_t getMask() const { return o2::detectors::DetID(detID).getMask(); }
  const char* getName() const { return o2::detectors::DetID::getName(detID); }
  uint32_t HBaccepted; /// Number of HB frames in TF to be accepted
  std::string mode = "";
  uint32_t ferst = 0;
  void printStream(std::ostream& stream) const;
  ClassDefNV(CTPDetector, 1)
};
/// List of detectors
struct CTPCluster {
  CTPCluster() = default;
  std::string name = "";
  uint32_t hwMask = 0;
  o2::detectors::DetID::mask_t maskCluster;
  std::string getClusterDetNames() const { return o2::detectors::DetID::getNames(maskCluster, ' '); }
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPCluster, 3)
};
/// Class = Mask+Descriptor+Cluster
struct CTPClass {
  CTPClass() = default;
  std::string name = "";
  std::uint64_t classMask = 0;
  CTPDescriptor const* descriptor = nullptr;
  CTPCluster const* cluster = nullptr;
  int clusterIndex = 0xff;
  int descriptorIndex = 0xff;
  uint32_t downScale = 1;
  std::vector<BCMask const*> BCClassMask;
  int getIndex() const { return ((classMask > 0) ? log2(classMask) : 0xff); }
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPClass, 4);
};
struct CTPInputsConfiguration {
  CTPInputsConfiguration() = default;
  std::vector<CTPInput> CTPInputs;
  static const std::vector<CTPInput> CTPInputsDefault;
  int createInputsConfigFromFile(std::string& filename);
  void printStream(std::ostream& strem) const;
  static CTPInputsConfiguration defaultInputConfig;
  static void initDefaultInputConfig();
  static std::string getInputNameFromIndex100(int index);
  static std::string getInputNameFromIndex(int index);
  static int getInputIndexFromName(std::string& name);
  ClassDefNV(CTPInputsConfiguration, 0);
};
class CTPConfiguration
{
 public:
  // static CTPInputsConfiguration mCTPInputsConfiguration;
  const static std::map<std::string, std::string> detName2LTG;
  CTPConfiguration() = default;
  bool isDetector(const o2::detectors::DetID& det);
  static void capitaliseString(std::string& str);
  static bool isNumber(const std::string& s);
  int addInput(std::string& inp, int clsindex, std::map<int, std::vector<int>>& descInputsIndex);
  enum ConfigPart { START,
                    VERSION,
                    RUN,
                    INPUTS,
                    MASKS,
                    GENS,
                    DESCRIPTORS,
                    LTG,
                    LTGitems,
                    CLUSTER,
                    CLASS,
                    UNKNOWN };
  int loadConfigurationRun3(const std::string& ctpconfiguartion);
  void printStream(std::ostream& stream) const;
  void setRunNumber(uint32_t runnumber) { mRunNumber = runnumber; }
  std::vector<CTPInput>& getCTPInputs() { return mInputs; }
  std::vector<CTPClass>& getCTPClasses() { return mCTPClasses; }
  const std::vector<CTPInput>& getCTPInputs() const { return mInputs; }      // Read-only interface
  const std::vector<CTPClass>& getCTPClasses() const { return mCTPClasses; } // Read-only interface
  uint64_t getInputMask(const std::string& name) const;
  int getInputIndex(const std::string& name) const;
  bool isMaskInInputs(const uint64_t& mask) const;
  bool isBCMaskInConfig(const std::string maskname) const;
  const BCMask* isBCMaskInConfigP(const std::string bcmask) const;
  const CTPInput* isInputInConfig(const std::string inpname) const;
  const CTPInput* isInputInConfig(const int index) const;
  const CTPDescriptor* isDescriptorInConfig(const std::string descname, int& index) const;
  void createInputsInDecriptorsFromNames();
  uint64_t getDecrtiptorInputsMask(const std::string& name) const;
  std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> getDet2InputMap();
  uint64_t getTriggerClassMask() const;
  std::vector<int> getTriggerClassList() const;
  uint32_t getRunNumber() { return mRunNumber; };
  std::vector<std::string> getDetectorList() const;
  o2::detectors::DetID::mask_t getDetectorMask() const;
  uint64_t getClassMaskForInputMask(uint64_t inputMask) const;
  void printConfigString() { std::cout << mConfigString << std::endl; };
  std::string getConfigString() { return mConfigString; };
  CTPDescriptor* getDescriptor(int index) { return &mDescriptors[index]; };
  int assignDescriptors();
  int checkConfigConsistency() const;

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
  int processConfigurationLineRun3(std::string& line, int& level, std::map<int, std::vector<int>>& descInputsIndex);
  int processConfigurationLineRun3v2(std::string& line, int& level, std::map<int, std::vector<int>>& descInputsIndex);
  ClassDefNV(CTPConfiguration, 6);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_CONFIGURATION_H_
