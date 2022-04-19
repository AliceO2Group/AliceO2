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

#ifndef _BASE_TFIDINFOHELPER_H_
#define _BASE_TFIDINFOHELPER_H_

#include "CommonDataFormat/TFIDInfo.h"

namespace o2
{
namespace framework
{
class ProcessingContext;
}

namespace base
{

struct TFIDInfoHelper {
  static void fillTFIDInfo(o2::framework::ProcessingContext& pc, o2::dataformats::TFIDInfo& ti);
};

} // namespace base
} // namespace o2

#endif
