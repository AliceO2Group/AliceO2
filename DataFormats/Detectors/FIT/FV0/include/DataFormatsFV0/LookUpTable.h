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
//
// file RawEventData.h class  for RAW data format
// Alla.Maevskaya
//  simple look-up table just to feed digits 2 raw procedure.
// Will be really set after module/electronics connections
//
#ifndef ALICEO2_FV0_LOOKUPTABLE_H_
#define ALICEO2_FV0_LOOKUPTABLE_H_
////////////////////////////////////////////////
// Look Up Table FV0
//////////////////////////////////////////////

#include "DataFormatsFIT/LookUpTable.h"
#include "CommonUtils/NameConf.h"

namespace o2
{
namespace fv0
{

namespace new_lut

{
// Singleton for LookUpTable
template <typename LUT>
class SingleLUT : public LUT
{
 private:
  SingleLUT(const std::string& ccdbPath, const std::string& ccdbPathToLUT) : LUT(ccdbPath, ccdbPathToLUT) {}
  SingleLUT(const std::string& pathToFile) : LUT(pathToFile) {}
  SingleLUT(const SingleLUT&) = delete;
  SingleLUT& operator=(SingleLUT&) = delete;

 public:
  static constexpr char sDetectorName[] = "FV0";
  static constexpr char sDefaultLUTpath[] = "FV0/Config/LookupTable";
  static constexpr char sObjectName[] = "LookupTable";
  inline static std::string sCurrentCCDBpath = "";
  inline static std::string sCurrentLUTpath = sDefaultLUTpath;
  // Before instance() call, setup url and path
  static void setCCDBurl(const std::string& url) { sCurrentCCDBpath = url; }
  static void setLUTpath(const std::string& path) { sCurrentLUTpath = path; }
  static SingleLUT& Instance()
  {
    if (sCurrentCCDBpath == "") {
      sCurrentCCDBpath = o2::base::NameConf::getCCDBServer();
    }
    static SingleLUT instanceLUT(sCurrentCCDBpath, sCurrentLUTpath);
    return instanceLUT;
  }
};
} // namespace new_lut

using SingleLUT = new_lut::SingleLUT<o2::fit::LookupTableBase<>>;

} // namespace fv0
} // namespace o2
#endif
