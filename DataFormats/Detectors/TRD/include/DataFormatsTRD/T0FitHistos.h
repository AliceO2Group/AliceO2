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

/// \file T0FitHistos.h
/// \brief Class to store the TRD PH values for each TRD chamber

#ifndef ALICEO2_T0FITHISTOS_H
#define ALICEO2_T0FITHISTOS_H

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/PHData.h"
#include "Rtypes.h"
#include <vector>
#include <memory>

namespace o2
{
namespace trd
{

class T0FitHistos
{
 public:
  T0FitHistos() = default;
  T0FitHistos(const T0FitHistos&) = default;
  ~T0FitHistos() = default;
  auto getDetector(int index) const { return mDet[index]; }
  auto getTimeBin(int index) const { return mTB[index]; }
  auto getADC(int index) const { return mADC[index]; }
  auto getNEntries() const { return mNEntriesTot; }

  void fill(const std::vector<o2::trd::PHData>& data);
  void merge(const T0FitHistos* prev);
  void print();

 private:
  std::vector<int> mDet{};
  std::vector<int> mTB{};
  std::vector<int> mADC{};
  size_t mNEntriesTot{0};

  ClassDefNV(T0FitHistos, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_T0FITHISTOS_H
