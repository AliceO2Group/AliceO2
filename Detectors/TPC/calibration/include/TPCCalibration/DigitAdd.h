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

/// \file DigitAdd.h
/// \brief Extension of TPC Digit adding draw query functions for simple drawing
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)

#ifndef ALICEO2_TPC_DIGITADD_H_
#define ALICEO2_TPC_DIGITADD_H_

#include "DataFormatsTPC/Digit.h"

namespace o2::tpc
{
/// \class Digit
class DigitAdd : public Digit
{
 public:
  int sector() const;
  float lx() const;
  float ly() const;
  float gx() const;
  float gy() const;
  float cpad() const;

  ClassDefNV(DigitAdd, 1)
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_DIGITADD_H_
