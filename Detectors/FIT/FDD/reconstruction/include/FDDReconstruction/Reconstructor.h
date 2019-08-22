// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CollisionTimeRecoTask.h
/// \brief Definition of the FDD reconstruction
#ifndef ALICEO2_FDD_RECONSTRUCTOR_H
#define ALICEO2_FDD_RECONSTRUCTOR_H

#include <vector>
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/RecPoint.h"
namespace o2
{
namespace fdd
{
class Reconstructor
{
 public:
  Reconstructor() = default;
  ~Reconstructor() = default;
  void Process(const o2::fdd::Digit& digit, RecPoint& recPoint) const;
  void Finish();

 private:
  ClassDefNV(Reconstructor, 1);
};
} // namespace fdd
} // namespace o2
#endif
