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

#ifndef ALICEO2_PRIMARYVERTEX_EXT_H
#define ALICEO2_PRIMARYVERTEX_EXT_H

#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

namespace o2
{
namespace dataformats
{

// extended primary vertex info

struct PrimaryVertexExt : public PrimaryVertex {
  using PrimaryVertex::PrimaryVertex;
  std::array<uint16_t, o2::dataformats::GlobalTrackID::Source::NSources> nSrc{}; // N contributors for each source type
  int VtxID = -1;                                                                // original vtx ID
  float FT0Amp = -1;                                                             // amplitude of closest FT0 trigger
  float FT0A = -1;                                                               // amplitude of the A side
  float FT0Time = -1.;                                                           // time of closest FT0 trigger

  int getNSrc(int i) const { return nSrc[i]; }

#ifndef GPUCA_ALIGPUCODE
  void print() const;
  std::string asString() const;
#endif

  ClassDefNV(PrimaryVertexExt, 2);
};

#ifndef GPUCA_ALIGPUCODE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::PrimaryVertexExt& v);
#endif

} // namespace dataformats

/// Defining PrimaryVertexExt explicitly as messageable
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dataformats::PrimaryVertexExt> : std::true_type {
};
} // namespace framework

} // namespace o2
#endif
