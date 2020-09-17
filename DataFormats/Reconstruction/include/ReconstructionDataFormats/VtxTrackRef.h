// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  VtxTrackRef.h
/// \brief Referenc on track indices contributing to the vertex, with possibility chose tracks from specific source (global, ITS, TPC...)
/// \author ruben.shahoyan@cern.ch

#ifndef O2_VERTEX_TRACK_REF
#define O2_VERTEX_TRACK_REF

#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include <cassert>
#include <array>
#include <iosfwd>
#include <string>

namespace o2
{
namespace dataformats
{

class VtxTrackRef : public RangeReference<int, int>
{
 public:
  VtxTrackRef(int ent, int n) : RangeReference(ent, n)
  {
    auto end = ent + n;
    for (int i = VtxTrackIndex::Source::NSources - 1; i--;) {
      mFirstEntrySource[i] = end; // only 1st source (base reference) is filled at constructor level
    }
  }
  using RangeReference<int, int>::RangeReference;
  void print() const;
  std::string asString() const;

  // get 1st of entry of indices for given source
  int getFirstEntryOfSource(int s) const
  {
    assert(s >= 0 && s < VtxTrackIndex::NSources);
    return s ? mFirstEntrySource[s - 1] : getFirstEntry();
  }

  // get number of entries for given source
  int getEntriesOfSource(int s) const
  {
    return (s == VtxTrackIndex::NSources - 1 ? (getFirstEntry() + getEntries()) : getFirstEntryOfSource(s + 1)) - getFirstEntryOfSource(s);
  }

  void setFirstEntryOfSource(int s, int i)
  {
    assert(s >= 0 && s < VtxTrackIndex::NSources);
    if (s) {
      mFirstEntrySource[s - 1] = i;
    } else {
      setFirstEntry(i);
    }
  }

 private:
  std::array<int, VtxTrackIndex::Source::NSources - 1> mFirstEntrySource{0};

  ClassDefNV(VtxTrackRef, 1);
};

std::ostream& operator<<(std::ostream& os, const o2::dataformats::VtxTrackRef& v);

} // namespace dataformats
} // namespace o2

#endif
