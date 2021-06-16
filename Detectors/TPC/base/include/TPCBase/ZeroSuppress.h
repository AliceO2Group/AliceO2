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

/// \file ZeroSuppressed.h
/// \brief Class for the TPC zero suppressed data format
/// \author Johannes Lehrbach

#ifndef ALICEO2_DATAFORMATSTPC_ZEROSUPPRESS_H
#define ALICEO2_DATAFORMATSTPC_ZEROSUPPRESS_H

#include <cmath>
#include <vector>
#include <array>

#include <iostream>

#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include <gsl/span>

namespace o2
{
namespace tpc
{
class ZeroSuppress
{

 private:
  std::vector<std::vector<ZeroSuppressedContainer8kb>> z0Pages = {}; //vector of 8kb pages as zero suppressed output
  Mapper& mapper;

 public:
  /// constructor
  ZeroSuppress() : mapper(Mapper::instance()){};
  /// destructor
  virtual ~ZeroSuppress() = default;
  ZeroSuppress(const ZeroSuppress&) = delete;

  void process();
  void DecodeZSPages(gsl::span<const ZeroSuppressedContainer8kb>* z0in, std::vector<Digit>* outDigits, int firstHBF);
};

} // namespace tpc
} // namespace o2

#endif // ALICEO2_DATAFORMATSTPC_ZEROSUPPRESS_H
