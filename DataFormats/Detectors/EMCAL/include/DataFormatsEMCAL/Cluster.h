// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_CLUSTER_H_
#define ALICEO2_EMCAL_CLUSTER_H_

#include <array>
#include <iosfwd>
#include <string>
#include <vector>
#include "CommonDataFormat/TimeStamp.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{

namespace emcal
{

/// \class Cluster
/// \brief EMCAL Cluster
///
class Cluster : public o2::dataformats::TimeStamp<Float16_t>
{
  using CellIndexRange = o2::dataformats::RangeRefComp<8>;

 public:
  Cluster() = default;
  Cluster(Float_t time, int firstcell, int ncells);
  ~Cluster() noexcept = default;

  Int_t getNCells() const { return mCellIndices.getEntries(); }
  Int_t getCellIndexFirst() const { return mCellIndices.getFirstEntry(); }
  CellIndexRange getCellIndexRange() const { return mCellIndices; }

  void setCellIndices(int firstcell, int ncells)
  {
    mCellIndices.setFirstEntry(firstcell);
    mCellIndices.setEntries(ncells);
  }
  void setCellIndexFirst(int firstcell) { mCellIndices.setFirstEntry(firstcell); }
  void setNCells(int ncells) { mCellIndices.setEntries(ncells); }

  void PrintStream(std::ostream& stream) const;

 private:
  CellIndexRange mCellIndices; ///< Cells contributing to a cluser
  ClassDefNV(Cluster, 1);
};

std::ostream& operator<<(std::ostream& stream, const o2::emcal::Cluster& cluster);

} // namespace emcal

} // namespace o2

#endif
