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

///
/// @file   IDCsVsSACs.h
/// @author Bhawani Singh
///

#ifndef AliceO2_TPC_IDCSVSSACS_H
#define AliceO2_TPC_IDCSVSSACS_H

// root includes
#include <string>

// o2 includes
#include "DataFormatsTPC/Defs.h"

class TCanvas;
namespace o2::tpc
{
template <class>
class IDCCCDBHelper;
template <class>
class SACCCDBHelper;
} // namespace o2::tpc
// namespace o2::tpc
namespace o2::tpc::qc
{
/// Keep QC information for SACs vs IDCs related observables
/// \tparam DataT the data type for the IDCDelta which are stored in the CCDB (unsigned short, unsigned char, float)
class IDCsVsSACs
{
 public:
  IDCsVsSACs() = default;
  IDCsVsSACs(IDCCCDBHelper<unsigned char>* mIDCs,
             SACCCDBHelper<unsigned char>* mSACs)
  {
    mCCDBHelper = mIDCs;
    mSacCCDBHelper = mSACs;
  }
  /// draw IDC0 and SAC0 side by side
  TCanvas* drawComparisionSACandIDCZero(TCanvas* outputCanvas, int nbins1D, float xMin1D, float xMax1D, int nbins1DSAC, float xMin1DSAC, float xMax1DSAC) const;

 private:
  IDCCCDBHelper<unsigned char>* mCCDBHelper;
  SACCCDBHelper<unsigned char>* mSacCCDBHelper;

  ClassDefNV(IDCsVsSACs, 1)
};
} // namespace o2::tpc::qc
#endif