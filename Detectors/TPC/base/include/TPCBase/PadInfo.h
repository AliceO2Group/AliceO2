// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef AliceO2_TPC_PadInfo_H
#define AliceO2_TPC_PadInfo_H

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/FECInfo.h"

namespace o2
{
namespace tpc
{

class PadInfo
{
 public:
 private:
  GlobalPadNumber mIndex{}; /// unique pad index in sector
  PadPos mPadPos{};         /// pad row and pad
  PadCentre mPadCentre{};   /// pad coordingate as seen for sector A04 in global ALICE coordiantes
  FECInfo mFECInfo{};       /// FEC mapping information
};

} // namespace tpc
} // namespace o2

#endif
