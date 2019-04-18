// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/PersistentVector.h
/// \brief  Persistent vector for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 April 2018
#ifndef O2_MID_PERSISTENTVECTOR_H
#define O2_MID_PERSISTENTVECTOR_H

#include <vector>
#include <gsl/gsl>

namespace o2
{
namespace mid
{
/// Class implementing a vector of contiguous objects which can be re-used
/// This saves time avoiding to allocate memory
template <class Tpl>
class PersistentVector
{
 public:
  /// Reserves size
  inline void reserve(size_t size) { mVector.reserve(size); }

  /// Gets size
  inline size_t size() const { return mVectorSize; }

  Tpl& next()
  {
    /// Gets next object in vector
    if (mVectorSize >= mVector.size()) {
      mVector.push_back(Tpl());
    }
    return mVector[mVectorSize++];
  }

  /// Start of vector
  inline typename std::vector<Tpl>::iterator begin() { return mVector.begin(); }

  /// End of vector
  inline typename std::vector<Tpl>::iterator end() { return mVector.begin() + mVectorSize; }

  /// Start of vector
  inline typename std::vector<Tpl>::const_iterator begin() const { return mVector.begin(); }

  /// End of vector
  inline typename std::vector<Tpl>::const_iterator end() const { return mVector.begin() + mVectorSize; }

  /// Gets last object in vector
  inline Tpl& back() { return mVector.back(); }

  /// Rewinds vector
  inline void rewind() { mVectorSize = 0; }

  /// Rewinds vector
  inline void rewind(size_t howMuch)
  {
    if (howMuch <= mVectorSize) {
      mVectorSize -= howMuch;
    }
  }

  /// Returns the object at idx
  inline Tpl& operator[](size_t idx) { return mVector[idx]; }

  /// Returns the const object at idx
  inline const Tpl& operator[](size_t idx) const { return mVector[idx]; }

  std::vector<Tpl> as_vector() const
  {
    /// Returns a copy of the vector with the proper size
    std::vector<Tpl> vec(mVector.begin(), mVector.begin() + mVectorSize);
    return std::move(vec);
  }

  gsl::span<const Tpl> as_span() const
  {
    /// Returns a span of the vector with the proper size
    gsl::span<const Tpl> vec(mVector.data(), mVectorSize);
    return std::move(vec);
  }

 private:
  std::vector<Tpl> mVector;
  size_t mVectorSize = 0;
};
} // namespace mid
} // namespace o2
#endif /* O2_MID_PERSISTENTVECTOR_H */
