// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ConstMCTruthContainer.h
/// \brief A const (ready only) version of MCTruthContainer
/// \author Sandro Wenzel - August 2020

#ifndef O2_CONSTMCTRUTHCONTAINER_H
#define O2_CONSTMCTRUTHCONTAINER_H

#include <SimulationDataFormat/MCTruthContainer.h>
#include <Framework/Traits.h>

namespace o2
{
namespace dataformats
{

/// @class ConstMCTruthContainer
/// @brief A read-only version of MCTruthContainer allowing for storage optimisation
///
/// This provides access functionality to MCTruthContainer with optimized linear storage
/// so that the data can easily be shared in memory or sent over network.
/// This container needs to be initialized by calling "flatten_to" from an existing
/// MCTruthContainer
template <typename TruthElement>
class ConstMCTruthContainer : public std::vector<char>
{
 public:
  // (unfortunately we need these constructors for DPL)
  using std::vector<char>::vector;
  ConstMCTruthContainer() = default;

  // const data access
  // get individual const "view" container for a given data index
  // the caller can't do modifications on this view
  MCTruthHeaderElement const& getMCTruthHeader(uint32_t dataindex) const
  {
    return getHeaderStart()[dataindex];
  }

  gsl::span<const TruthElement> getLabels(uint32_t dataindex) const
  {
    if (dataindex >= getIndexedSize()) {
      return gsl::span<const TruthElement>();
    }
    const auto start = getMCTruthHeader(dataindex).index;
    const auto labelsptr = getLabelStart();
    return gsl::span<const TruthElement>(&labelsptr[start], getSize(dataindex));
  }

  // return the number of original data indexed here
  size_t getIndexedSize() const { return size() >= sizeof(FlatHeader) ? getHeader().nofHeaderElements : 0; }

  // return the number of labels  managed in this container
  size_t getNElements() const { return size() >= sizeof(FlatHeader) ? getHeader().nofTruthElements : 0; }

 private:
  using FlatHeader = typename MCTruthContainer<TruthElement>::FlatHeader;

  size_t getSize(uint32_t dataindex) const
  {
    // calculate size / number of labels from a difference in pointed indices
    const auto size = (dataindex < getIndexedSize() - 1)
                        ? getMCTruthHeader(dataindex + 1).index - getMCTruthHeader(dataindex).index
                        : getNElements() - getMCTruthHeader(dataindex).index;
    return size;
  }

  /// Restore internal vectors from a raw buffer
  /// The two vectors are resized according to the information in the \a FlatHeader
  /// struct at the beginning of the buffer. Data is copied to the vectors.
  TruthElement const* const getLabelStart() const
  {
    auto* source = &(*this)[0];
    auto flatheader = getHeader();
    source += sizeof(FlatHeader);
    const size_t headerSize = flatheader.sizeofHeaderElement * flatheader.nofHeaderElements;
    source += headerSize;
    return (TruthElement const* const)source;
  }

  FlatHeader const& getHeader() const
  {
    const auto* source = &(*this)[0];
    const auto& flatheader = *reinterpret_cast<FlatHeader const*>(source);
    return flatheader;
  }

  MCTruthHeaderElement const* const getHeaderStart() const
  {
    auto* source = &(*this)[0];
    source += sizeof(FlatHeader);
    return (MCTruthHeaderElement const* const)source;
  }
};
} // namespace dataformats
} // namespace o2

// This is done so that DPL treats this container as a vector.
// In particular in enables
// a) --> snapshot without ROOT dictionary (as a flat buffer)
// b) --> requesting the resource in shared mem using make<T>
namespace o2::framework
{
template <typename T>
struct is_specialization<o2::dataformats::ConstMCTruthContainer<T>, std::vector> : std::true_type {
};
} // namespace o2::framework

namespace o2
{
namespace dataformats
{

// A "view" label container without owning the storage (similar to gsl::span)
template <typename TruthElement>
class ConstMCTruthContainerView
{
 public:
  ConstMCTruthContainerView(gsl::span<const char> const bufferview) : mStorage(bufferview){};

  // const data access
  // get individual const "view" container for a given data index
  // the caller can't do modifications on this view
  MCTruthHeaderElement const& getMCTruthHeader(uint32_t dataindex) const
  {
    return getHeaderStart()[dataindex];
  }

  gsl::span<const TruthElement> getLabels(uint32_t dataindex) const
  {
    if (dataindex >= getIndexedSize()) {
      return gsl::span<const TruthElement>();
    }
    const auto start = getMCTruthHeader(dataindex).index;
    const auto labelsptr = getLabelStart();
    return gsl::span<const TruthElement>(&labelsptr[start], getSize(dataindex));
  }

  // return the number of original data indexed here
  size_t getIndexedSize() const { return mStorage.size() >= sizeof(FlatHeader) ? getHeader().nofHeaderElements : 0; }

  // return the number of labels  managed in this container
  size_t getNElements() const { return mStorage.size() >= sizeof(FlatHeader) ? getHeader().nofTruthElements : 0; }

 private:
  gsl::span<const char> mStorage;

  using FlatHeader = typename MCTruthContainer<TruthElement>::FlatHeader;

  size_t getSize(uint32_t dataindex) const
  {
    // calculate size / number of labels from a difference in pointed indices
    const auto size = (dataindex < getIndexedSize() - 1)
                        ? getMCTruthHeader(dataindex + 1).index - getMCTruthHeader(dataindex).index
                        : getNElements() - getMCTruthHeader(dataindex).index;
    return size;
  }

  /// Restore internal vectors from a raw buffer
  /// The two vectors are resized according to the information in the \a FlatHeader
  /// struct at the beginning of the buffer. Data is copied to the vectors.
  TruthElement const* const getLabelStart() const
  {
    auto* source = &(mStorage)[0];
    auto flatheader = getHeader();
    source += sizeof(FlatHeader);
    const size_t headerSize = flatheader.sizeofHeaderElement * flatheader.nofHeaderElements;
    source += headerSize;
    return (TruthElement const* const)source;
  }

  FlatHeader const& getHeader() const
  {
    const auto* source = &(mStorage)[0];
    const auto& flatheader = *reinterpret_cast<FlatHeader const*>(source);
    return flatheader;
  }

  MCTruthHeaderElement const* const getHeaderStart() const
  {
    auto* source = &(mStorage)[0];
    source += sizeof(FlatHeader);
    return (MCTruthHeaderElement const* const)source;
  }
};

} // namespace dataformats
} // namespace o2

#endif //O2_CONSTMCTRUTHCONTAINER_H
