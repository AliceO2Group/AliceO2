// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#ifndef O2_MCH_MAPPING_PADGROUPTYPE_H
#define O2_MCH_MAPPING_PADGROUPTYPE_H

#include <vector>
#include <ostream>

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl4
{

struct PadGroupType {
  PadGroupType(int nofPadsX, int nofPadsY, std::vector<int> ids);

  int getNofPads() const { return mNofPads; }

  int getNofPadsX() const { return mNofPadsX; }

  int getNofPadsY() const { return mNofPadsY; }

  int fastIndex(int ix, int iy) const { return ix + iy * mNofPadsX; }

  int id(int fastIndex) const;

  /// Return the index of the pad with indices = (ix,iy)
  /// or -1 if not found
  int id(int ix, int iy) const { return id(fastIndex(ix, iy)); }

  int iy(int fastIndex) const { return fastIndex / mNofPadsX; }

  int ix(int fastIndex) const { return fastIndex - iy(fastIndex) * mNofPadsX; }

  std::vector<int> fastIndices() const { return mFastIndices; }

  /// Whether pad with given id exists
  bool hasPadById(int id) const;

  friend std::ostream& operator<<(std::ostream& os, const PadGroupType& type);

  int getIndex(int ix, int iy) const;

  std::vector<int> mFastId;
  std::vector<int> mFastIndices;
  int mNofPads;
  int mNofPadsX;
  int mNofPadsY;
};

PadGroupType getPadGroupType(int i);

} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2

#endif
