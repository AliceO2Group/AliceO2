// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digits2Raw.h
/// \brief converts digits to raw format
/// \author pietro.cortese@cern.ch

#ifndef ALICEO2_ZDC_DIGITS2RAW_H_
#define ALICEO2_ZDC_DIGITS2RAW_H_
#include <string>
#include <Rtypes.h>
#include "DataFormatsZDC/RawEventData.h"
#include "ZDCBase/ModuleConfig.h"

namespace o2
{
namespace zdc
{
class Digits2Raw
{
 public:
  Digits2Raw() = default;
  void readDigits(const std::string& outDir, const std::string& fileDigitsName);
  void convertDigits(int ibc);
  void setModuleConfig(const ModuleConfig *moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };

 private:
  void setTriggerMask();
  std::vector<o2::zdc::BCData> mzdcBCData, *mzdcBCDataPtr = &mzdcBCData;
  std::vector<o2::zdc::ChannelData> mzdcChData, *mzdcChDataPtr = &mzdcChData;
  EventData mzdcData;
  const ModuleConfig* mModuleConfig = 0;

  UShort_t scalers[NModules][NChPerModule] = {0}; /// ZDC orbit scalers
  UShort_t last_bc = 0;
  UInt_t last_orbit = 0;
  uint32_t mTriggerMask = 0;
  std::string mPrintTriggerMask="";

  /////////////////////////////////////////////////
  ClassDefNV(Digits2Raw, 1);
};
} // namespace zdc
} // namespace o2

#endif
