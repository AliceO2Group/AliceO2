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

#ifndef O2_GLOBALOFFSETSINFOOBJECT_H
#define O2_GLOBALOFFSETSINFOOBJECT_H

#include "Rtypes.h"

namespace o2
{
namespace ft0
{
class GlobalOffsetsInfoObject
{
 public:
  GlobalOffsetsInfoObject(short t0A, short t0C, short t0AC) : mT0A(t0A), mT0C(t0C), mT0AC(t0AC);
  GlobalOffsetsInfoObject() = default;
  ~GlobalOffsetsInfoObject() = default;

  void setT0A(short t0A) { mT0A = t0A; }
  short getT0A() const { return mT0A; }
  void setT0C(short t0C) { mT0C = t0C; }
  short getT0A() const { return mT0A; }
  void setT0AC(short t0AC) { mT0AC = t0AC; }
  short getT0AC() const { return mT0AC; }

 private:
  short mT0A;
  short mT0C;
  short mT0AC;

  ClassDefNV(GlobalOffsetsInfoObject, 1);
};
} // namespace ft0
} // namespace o2

#endif //O2_GLOBALOFFSETSINFOOBJECT_H
