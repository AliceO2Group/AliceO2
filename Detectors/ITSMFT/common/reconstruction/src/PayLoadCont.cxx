// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PayLoadCont.cxx
/// \brief Implementation of class for continuos buffer of ALPIDE data

#include "ITSMFTReconstruction/PayLoadCont.h"

using namespace o2::itsmft;

constexpr size_t PayLoadCont::MinCapacity;

void PayLoadCont::expand(size_t sz)
{
  ///< increase the buffer size
  auto* oldHead = mBuffer.data();
  if (sz < MinCapacity) {
    sz = MinCapacity;
  }
  if (sz < mBuffer.size()) { // never decrease the size
    return;
  }
  mBuffer.resize(sz);
  if (oldHead) { // fix the pointers to account for the reallocation
    int64_t diff = mBuffer.data() - oldHead;
    mPtr += diff;
    mEnd += diff;
  } else { // new buffer
    clear();
  }
}
