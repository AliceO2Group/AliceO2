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

  const InteractionRecord& getIR() const { return mIR; }
  void setIR(const InteractionRecord& ir) { mIR = ir; }

#ifndef ALIGPU_GPUCODE
  void print() const;
  std::string asString() const;
#endif

 protected:
  InteractionRecord mIR{}; ///< by default not assigned!

  ClassDefNV(PrimaryVertex, 1);
};

#ifndef ALIGPU_GPUCODE
std::ostream& operator<<(std::ostream& os, const o2::dataformats::PrimaryVertex& v);
#endif

} // namespace dataformats
} // namespace o2
#endif
