// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
  std::string mName;
  std::bitset<o2::constants::lhc::LHCMaxBunches> mBCmask;
  ClassDefNV(BCMask, 1);
};
struct CTPInput {
  CTPInput() = default;
  std::string mName;
  uint_fast64_t mInputMask;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPInput, 1);
};
struct CTPDescriptor {
  CTPDescriptor() = default;
  std::string mName;
  uint64_t mInputsMask;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPDescriptor, 1)
};
struct LTG {
  LTG() = default;
  std::string mName;
};
struct CTPCluster {
  CTPCluster() = default;
  std::string mName;
  ClassDefNV( CTPCluster, 1)
};
struct CTPClass {
  CTPClass() = default;
  std::string mName;
  uint64_t mClassMask;
  CTPDescriptor* mDescriptor;
  void printStream(std::ostream& strem) const;
  ClassDefNV(CTPClass, 1);
};
class CTPConfiguration {
  public:
    CTPConfiguration() = default;
    void addCTPClass(CTPClass & ctpclass);
  private:
    std::vector<CTPInput> Inputs;
    std::vector<CTPDescriptor> Descriptors;
    std::vector<CTPClass> CTPClasses;
    ClassDefNV(CTPConfiguration, 1);
};
} // namespace ctp
} // namespace o2
#endif //_CTP_CONFIGURATION_H_
