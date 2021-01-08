// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PRIMARYVERTEX_H
#define ALICEO2_PRIMARYVERTEX_H

#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/Vertex.h"

namespace o2
{
namespace dataformats
{

// primary vertex class: position, time with error + IR (by default: not assigned)

class PrimaryVertex : public Vertex<TimeStampWithError<float, float>>
{
 public:
  using Vertex<TimeStampWithError<float, float>>::Vertex;
  PrimaryVertex() = default;
  PrimaryVertex(const PrimaryVertex&) = default;
  ~PrimaryVertex() = default;

  const InteractionRecord& getIRMax() const { return mIRMax; }
  void setIRMax(const InteractionRecord& ir) { mIRMax = ir; }
  const InteractionRecord& getIRMin() const { return mIRMin; }
  void setIRMin(const InteractionRecord& ir) { mIRMin = ir; }
  void setIR(const InteractionRecord& ir) { mIRMin = mIRMax = ir; }
  bool hasUniqueIR() const { return !mIRMin.isDummy() && (mIRMin == mIRMax); }

#ifndef GPUCA_ALIGPUCODE
  void print() const;
  std::string asString() const;
#endif

 protected:
  InteractionRecord mIRMin{}; ///< by default not assigned!
  InteractionRecord mIRMax{}; ///< by default not assigned!

  ClassDefNV(PrimaryVertex, 1);
};

#ifndef GPUCA_ALIGPUCODE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::PrimaryVertex& v);
#endif

} // namespace dataformats

/// Defining PrimaryVertex explicitly as messageable
namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dataformats::PrimaryVertex> : std::true_type {
};
} // namespace framework

} // namespace o2
#endif
