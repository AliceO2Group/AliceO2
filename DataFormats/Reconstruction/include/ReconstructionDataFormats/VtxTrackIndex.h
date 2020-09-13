// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  VtxTrackIndex.h
/// \brief Index of track attached to vertx: index in its proper container, container source and flags
/// \author ruben.shahoyan@cern.ch

#ifndef O2_VERTEX_TRACK_INDEX
#define O2_VERTEX_TRACK_INDEX

#include "CommonDataFormat/AbstractRef.h"
#include <iosfwd>
#include <string>

namespace o2
{
namespace dataformats
{

class VtxTrackIndex : public AbstractRef<26, 3, 3>
{
 public:
  enum Source : uint8_t { // provenance of the track
    TPCITS,
    ITS,
    TPC,
    NSources
  };
  enum Flags : uint8_t {
    Contributor, // flag that it contributes to vertex fit
    Reserved,    //
    Ambiguous,   // flag that attachment is ambiguous
    NFlags
  };

  using AbstractRef<26, 3, 3>::AbstractRef;

  void print() const;
  std::string asString() const;

  ClassDefNV(VtxTrackIndex, 1);
};

std::ostream& operator<<(std::ostream& os, const o2::dataformats::VtxTrackIndex& v);

} // namespace dataformats
} // namespace o2

#endif
