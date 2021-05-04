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
namespace o2
{
namespace ctp
{
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
  o2::detectors::DetID::ID detID;
  std::uint64_t inputMask;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPInput, 1);
};
struct CTPDescriptor {
  CTPDescriptor() = default;
  std::string name;
  std::uint64_t inputsMask;
  std::vector<std::string> inputsNames;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPDescriptor, 1)
};
/// The main part is Local Trigger Generator (LTG)
struct CTPDetector {
  CTPDetector() = default;
  o2::detectors::DetID::ID detID;
  uint32_t HBaccepted; /// Number of HB frames in TF to be accepted
  void printStream(std::ostream& strem) const;
};
struct CTPCluster {
  CTPCluster() = default;
  std::string name;
  uint32_t detectorsMask;
  std::vector<std::string> detectorsNames;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPCluster, 1)
};
struct CTPClass {
  CTPClass() = default;
  std::string name;
  std::uint64_t classMask;
  CTPDescriptor* descriptor;
  CTPCluster* cluster;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPClass, 1);
};
class CTPConfiguration
{
 public:
  CTPConfiguration() = default;
  void addBCMask(const BCMask& bcmask);
  void addCTPInput(const CTPInput& input);
  void addCTPDescriptor(const CTPDescriptor& descriptor);
  void addCTPDetector(const CTPDetector& detector);
  void addCTPCluster(const CTPCluster& cluster);
  void addCTPClass(const CTPClass& ctpclass);
  void printStream(std::ostream& stream) const;

 private:
  std::string mName;
  std::vector<BCMask> mBCMasks;
  std::vector<CTPInput> mInputs;
  std::vector<CTPDescriptor> mDescriptors;
  std::vector<CTPDetector> mDetectors;
  std::vector<CTPCluster> mClusters;
  std::vector<CTPClass> mCTPClasses;
  ClassDefNV(CTPConfiguration, 1);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_CONFIGURATION_H_
