// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPCSECTORHEADER_H
#define O2_TPCSECTORHEADER_H

#include "Headers/DataHeader.h"

namespace o2
{
namespace tpc
{

struct TPCSectorHeader : public o2::header::BaseHeader {
  // Required to do the lookup
  static const o2::header::HeaderType sHeaderType;
  static const uint32_t sVersion = 1;

  TPCSectorHeader(int s)
    : BaseHeader(sizeof(TPCSectorHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), sector(s)
  {
  }

  int sector;
};
} // namespace tpc
} // namespace o2

#endif // O2_TPCSECTORHEADER_H
