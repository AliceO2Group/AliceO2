// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <string>
#include <vector>
#include <bitset>
#include <map>
namespace o2
{
namespace ctp
{
/// Database constants
const std::string CCDBPathCTPConfig = "CTP/Config";
///
bool isDetector(o2::detectors::DetID& det);
/// CTP Config items
struct BCMask {
  BCMask() = default;
  std::string name;
  std::bitset<o2::constants::lhc::LHCMaxBunches> BCmask;
  void printStream(std::ostream& strem) const;
  ClassDefNV(BCMask, 1);
};
struct CTPInput {
  CTPInput() = default;
  std::string name;
  std::string level;
  std::uint64_t inputMask;
  o2::detectors::DetID::ID detID;
  std::string getInputDetName() const { return o2::detectors::DetID::getName(detID); }
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPInput, 2);
};
struct CTPDescriptor {
  CTPDescriptor() = default;
  std::string name;
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
  void printStream(std::ostream& strem) const;
};
struct CTPCluster {
  CTPCluster() = default;
  std::string name;
  o2::detectors::DetID::mask_t maskCluster;
  std::string getClusterDetNames() const { return o2::detectors::DetID::getNames(maskCluster, ' '); }
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPCluster, 2)
};
struct CTPClass {
  CTPClass() = default;
  std::string name;
  std::uint64_t classMask;
  CTPDescriptor const* descriptor;
  CTPCluster const* cluster;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPClass, 1);
};
class CTPConfiguration
{
 public:
  CTPConfiguration() = default;
  bool isDetector(const o2::detectors::DetID& det);
  int loadConfiguration(const std::string& ctpconfiguartion);
  void addBCMask(const BCMask& bcmask);
  void addCTPInput(const CTPInput& input);
  void addCTPDescriptor(const CTPDescriptor& descriptor);
  void addCTPDetector(const CTPDetector& detector);
  void addCTPCluster(const CTPCluster& cluster);
  void addCTPClass(const CTPClass& ctpclass);
  void printStream(std::ostream& stream) const;
  std::vector<CTPInput>& getCTPInputs() { return mInputs; }
  std::vector<CTPClass>& getCTPClasses() { return mCTPClasses; }
  uint64_t getInputMask(const std::string& name);
  bool isMaskInInputs(const uint64_t& mask) const;
  CTPInput* isInputInConfig(const std::string inpname);
  uint64_t getDecrtiptorInputsMask(const std::string& name) const;
  std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> getDet2InputMap();

 private:
  std::string mName;
  std::string mVersion;
  std::vector<BCMask> mBCMasks;
  std::vector<CTPInput> mInputs;
  std::vector<CTPDescriptor> mDescriptors;
  std::vector<CTPDetector> mDetectors;
  std::vector<CTPCluster> mClusters;
  std::vector<CTPClass> mCTPClasses;
  int processConfigurationLine(std::string& line, int& level);
  ClassDefNV(CTPConfiguration, 2);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_CONFIGURATION_H_
