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

#ifndef O2_RECOCALIBINFOOBJECT_H
#define O2_RECOCALIBINFOOBJECT_H

#include "Rtypes.h"

namespace o2
{
namespace ft0
{
class RecoCalibInfoObject
{
 public:
  RecoCalibInfoObject(short t0a, short t0c, short t0ac) : mT0A(t0a), mT0C(t0c), mT0AC(t0ac){};
  RecoCalibInfoObject() = default;
  ~RecoCalibInfoObject() = default;

  void setT0A(short time) { mT0A = time; }
  void setT0C(short time) { mT0C = time; }
  void setT0AC(short time) { mT0AC = time; }
  [[nodiscard]] short getT0A() const { return mT0A; }
  [[nodiscard]] short getT0C() const { return mT0C; }
  [[nodiscard]] short getT0AC() const { return mT0AC; }

 private:
  short mT0A;
  short mT0C;
  short mT0AC;

  ClassDefNV(RecoCalibInfoObject, 1);
};
} // namespace ft0
} // namespace o2

#endif //O2_RecoCalibINFOOBJECT_H
