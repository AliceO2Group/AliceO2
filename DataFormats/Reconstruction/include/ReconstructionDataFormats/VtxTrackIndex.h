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
/// \brief Extention of GlobalTrackID by flags relevant for verter-track association
/// \author ruben.shahoyan@cern.ch

#ifndef O2_VERTEX_TRACK_INDEX
#define O2_VERTEX_TRACK_INDEX

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include <iosfwd>
#include <string>
#include <array>
#include <string_view>

namespace o2
{
namespace dataformats
{

class VtxTrackIndex : public GlobalTrackID
{
 public:
  enum Flags : uint8_t {
    Contributor, // flag that it contributes to vertex fit
    Reserved,    //
    Ambiguous,   // flag that attachment is ambiguous
    NFlags
  };

  using GlobalTrackID::GlobalTrackID;

  bool isPVContributor() const { return testBit(Contributor); }
  void setPVContributor() { setBit(Contributor); }

  bool isAmbiguous() const { return testBit(Ambiguous); }
  void setAmbiguous() { setBit(Ambiguous); }

  ClassDefNV(VtxTrackIndex, 2);
};

} // namespace dataformats
} // namespace o2

#endif
