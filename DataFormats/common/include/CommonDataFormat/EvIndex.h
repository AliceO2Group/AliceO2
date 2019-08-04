// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EvIndex.h
/// \brief Class to store event ID and index in the event for objects like track, cluster...
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_EVINDEX_H
#define ALICEO2_EVINDEX_H

#include <Rtypes.h>

namespace o2
{
namespace dataformats
{
// Composed Label to encode object origin in the tree or other segmented input

template <typename E = int, typename I = int>
class EvIndex
{
 public:
  EvIndex(E ev, I idx) { set(ev, idx); }
  EvIndex(const EvIndex<E, I>& src) = default;
  EvIndex() = default;
  ~EvIndex() = default;
  void set(E ev, I idx)
  {
    mEvent = ev;
    mIndex = idx;
  }
  E getEvent() const { return mEvent; }
  I getIndex() const { return mIndex; }
  void setEvent(E ev) { mEvent = ev; }
  void setIndex(I ind) { mIndex = ind; }
  void shiftEvent(E inc) { mEvent += inc; }
  void shiftIndex(I inc) { mIndex += inc; }

  void clear()
  {
    mEvent = 0;
    mIndex = 0;
  }

 private:
  E mEvent = 0; ///< ID of event or chunk or message containing referred object
  I mIndex = 0; ///< index in the event

  ClassDefNV(EvIndex, 1);
};
} // namespace dataformats
} // namespace o2

#endif
