// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MCTruthContainer.h
/// \brief Definition of a container to keep Monte Carlo truth external to simulation objects
/// \author Sandro Wenzel - June 2017

#ifndef ALICEO2_DATAFORMATS_MCTRUTH_H_
#define ALICEO2_DATAFORMATS_MCTRUTH_H_

#include <TNamed.h> // to have the ClassDef macros
#include <cstdint>  // uint8_t etc
#include <cassert>
#include <stdexcept>
#include <gsl/gsl> // for guideline support library; array_view
#include <type_traits>
#include <cstring> // memmove, memcpy
#include <memory>
#include <vector>
// type traits are needed for the compile time consistency check
// maybe to be moved out of Framework first
//#include "Framework/TypeTraits.h"

namespace o2
{
namespace dataformats
{
/// @struct MCTruthHeaderElement
/// @brief Simple struct having information about truth elements for particular indices:
/// how many associations we have and where they start in the storage
struct MCTruthHeaderElement {
  MCTruthHeaderElement() = default; // for ROOT IO

  MCTruthHeaderElement(uint i) : index(i) {}
  uint index = -1; // the index into the actual MC track storage (-1 if invalid)
  ClassDefNV(MCTruthHeaderElement, 1);
};

/// @class MCTruthContainer
/// @brief A container to hold and manage MC truth information/labels.
///
/// The actual MCtruth type is a generic template type and can be supplied by the user
/// It is meant to manage associations from one "dataobject" identified by an index into an array
/// to multiple TruthElements. Each "dataobject" is identified by a sequential index. Truth elements
/// belonging to one object are always in contingous sequence in the truth element storage. Since
/// multiple truth elements can be associated with one object, the header array stores the start
/// of the associated truth element sequence.
///
/// Since the class contains two subsequent vectors, it is not POD even if the TruthElement is
/// POD. ROOT serialization is rather inefficient and in addition has a large memory footprint
/// if the container has lots of (>1000000) elements. between 3 and 4x more than the actual
/// size is allocated. If the two vectors are flattend to a raw vector before streaming, the
/// serialization works without memory overhead. The deflate/inflate methods are called from
/// a custom streamer, storing the vectors in the raw buffer and vice versa, each of the methods
/// emptying the source data.
///
/// TODO:
/// - add move assignment from a source vector, by that passing an object which has access to
///   different underlying memory resources, until that, the pmr::MemoryResource has been
///   removed again
/// - add interface to access header and truth elements directly from the raw buffer, by that
///   inflation can be postponed until new elements are added, with the effect that inflation
///   can be avoided in most cases
///
/// Note:
/// The two original vector members could be transient, however reading serialized version 1
/// objects does not work correctly. In a different approach, the two vectors have been removed
/// completely with an efficient interface to the binary buffer, but the read pragma was not able
/// to access the member offset from the StreamerInfo.
template <typename TruthElement>
class MCTruthContainer
{
 private:
  // for the moment we require the truth element to be messageable in order to simply flatten the object
  // if it turnes out that other types are required this needs to be extended and method flatten nees to
  // be conditionally added
  // TODO: activate this check
  //static_assert(o2::framework::is_messageable<TruthElement>::value, "truth element type must be messageable");

  std::vector<MCTruthHeaderElement> mHeaderArray; // the header structure array serves as an index into the actual storage
  std::vector<TruthElement> mTruthArray;          // the buffer containing the actual truth information
  /// buffer used only for streaming the to above vectors in a flat structure
  /// TODO: use polymorphic allocator so that it can work on an underlying custom memory resource,
  /// e.g. directly on the memory of the incoming message.
  std::vector<char> mStreamerData; // buffer used for streaming a flat raw buffer

  size_t getSize(uint dataindex) const
  {
    // calculate size / number of labels from a difference in pointed indices
    const auto size = (dataindex < getIndexedSize() - 1)
                        ? getMCTruthHeader(dataindex + 1).index - getMCTruthHeader(dataindex).index
                        : getNElements() - getMCTruthHeader(dataindex).index;
    return size;
  }

 public:
  // constructor
  MCTruthContainer() = default;
  // destructor
  ~MCTruthContainer() = default;
  // copy constructor
  MCTruthContainer(const MCTruthContainer& other) = default;
  // move constructor
  MCTruthContainer(MCTruthContainer&& other) = default;
  // assignment operator
  MCTruthContainer& operator=(const MCTruthContainer& other) = default;
  // move assignment operator
  MCTruthContainer& operator=(MCTruthContainer&& other) = default;

  using self_type = MCTruthContainer<TruthElement>;
  struct FlatHeader {
    uint8_t version = 1;
    uint8_t sizeofHeaderElement = sizeof(MCTruthHeaderElement);
    uint8_t sizeofTruthElement = sizeof(TruthElement);
    uint8_t reserved = 0;
    uint32_t nofHeaderElements;
    uint32_t nofTruthElements;
  };

  // access
  MCTruthHeaderElement const& getMCTruthHeader(uint dataindex) const { return mHeaderArray[dataindex]; }
  // access the element directly (can be encapsulated better away)... needs proper element index
  // which can be obtained from the MCTruthHeader startposition and size
  TruthElement const& getElement(uint elementindex) const { return mTruthArray[elementindex]; }
  // return the number of original data indexed here
  size_t getIndexedSize() const { return mHeaderArray.size(); }
  // return the number of elements managed in this container
  size_t getNElements() const { return mTruthArray.size(); }

  // get individual "view" container for a given data index
  // the caller can do modifications on this view (such as sorting)
  gsl::span<TruthElement> getLabels(uint dataindex)
  {
    if (dataindex >= getIndexedSize()) {
      return gsl::span<TruthElement>();
    }
    return gsl::span<TruthElement>(&mTruthArray[getMCTruthHeader(dataindex).index], getSize(dataindex));
  }

  // get individual const "view" container for a given data index
  // the caller can't do modifications on this view
  gsl::span<const TruthElement> getLabels(uint dataindex) const
  {
    if (dataindex >= getIndexedSize()) {
      return gsl::span<const TruthElement>();
    }
    return gsl::span<const TruthElement>(&mTruthArray[getMCTruthHeader(dataindex).index], getSize(dataindex));
  }

  void clear()
  {
    mHeaderArray.clear();
    mTruthArray.clear();
  }

  // add element for a particular dataindex
  // at the moment only strictly consecutive modes are supported
  void addElement(uint dataindex, TruthElement const& element)
  {
    if (dataindex < mHeaderArray.size()) {
      // look if we have something for this dataindex already
      // must currently be the last one
      if (dataindex != (mHeaderArray.size() - 1)) {
        throw std::runtime_error("MCTruthContainer: unsupported code path");
      }
    } else {
      // assert(dataindex == mHeaderArray.size());

      // add empty holes
      int holes = dataindex - mHeaderArray.size();
      assert(holes >= 0);
      for (int i = 0; i < holes; ++i) {
        mHeaderArray.emplace_back(mTruthArray.size());
      }
      // add a new one
      mHeaderArray.emplace_back(mTruthArray.size());
    }
    auto& header = mHeaderArray[dataindex];
    mTruthArray.emplace_back(element);
  }

  // convenience interface to add multiple labels at once
  // can use elements of any assignable type or sub-type
  template <typename CompatibleLabel>
  void addElements(uint dataindex, gsl::span<CompatibleLabel> elements)
  {
    static_assert(std::is_same<TruthElement, CompatibleLabel>::value ||
                    std::is_assignable<TruthElement, CompatibleLabel>::value ||
                    std::is_base_of<TruthElement, CompatibleLabel>::value,
                  "Need to add compatible labels");
    for (auto& e : elements) {
      addElement(dataindex, e);
    }
  }

  template <typename CompatibleLabel>
  void addElements(uint dataindex, const std::vector<CompatibleLabel>& v)
  {
    using B = typename std::remove_const<CompatibleLabel>::type;
    auto s = gsl::span<CompatibleLabel>(const_cast<B*>(&v[0]), v.size());
    addElements(dataindex, s);
  }

  // Add element at last position or for a previous index
  // (at random access position).
  // This might be a slow process since data has to be moved internally
  // so this function should be used with care.
  void addElementRandomAccess(uint dataindex, TruthElement const& element)
  {
    if (dataindex >= mHeaderArray.size()) {
      // a new dataindex -> push element at back

      // we still forbid to leave holes
      assert(dataindex == mHeaderArray.size());

      mHeaderArray.resize(dataindex + 1);
      mHeaderArray[dataindex] = mTruthArray.size();
      mTruthArray.emplace_back(element);
    } else {
      // if appending at end use fast function
      if (dataindex == mHeaderArray.size() - 1) {
        addElement(dataindex, element);
        return;
      }

      // existing dataindex
      // have to:
      // a) move data;
      // b) insert new element;
      // c) adjust indices of all headers right to this
      auto currentindex = mHeaderArray[dataindex].index;
      auto lastindex = currentindex + getSize(dataindex);
      assert(currentindex >= 0);

      // resize truth array
      mTruthArray.resize(mTruthArray.size() + 1);
      // move data (have to do from right to left)
      for (int i = mTruthArray.size() - 1; i > lastindex; --i) {
        mTruthArray[i] = mTruthArray[i - 1];
      }
      // insert new element
      mTruthArray[lastindex] = element;

      // fix headers
      for (uint i = dataindex + 1; i < mHeaderArray.size(); ++i) {
        auto oldindex = mHeaderArray[i].index;
        mHeaderArray[i].index = (oldindex != -1) ? oldindex + 1 : oldindex;
      }
    }
  }

  // merge another container to the back of this one
  void mergeAtBack(MCTruthContainer<TruthElement> const& other)
  {
    const auto oldtruthsize = mTruthArray.size();
    const auto oldheadersize = mHeaderArray.size();

    // copy from other
    std::copy(other.mHeaderArray.begin(), other.mHeaderArray.end(), std::back_inserter(mHeaderArray));
    std::copy(other.mTruthArray.begin(), other.mTruthArray.end(), std::back_inserter(mTruthArray));

    // adjust information of newly attached part
    for (uint i = oldheadersize; i < mHeaderArray.size(); ++i) {
      mHeaderArray[i].index += oldtruthsize;
    }
  }

  /// Flatten the internal arrays to the provided container
  /// Copies the content of the two vectors of PODs to a contiguous container.
  /// The flattened data starts with a specific header @ref FlatHeader describing
  /// size and content of the two vectors within the raw buffer.
  template <typename ContainerType>
  size_t flatten_to(ContainerType& container)
  {
    size_t bufferSize = sizeof(FlatHeader) + sizeof(MCTruthHeaderElement) * mHeaderArray.size() + sizeof(TruthElement) * mTruthArray.size();
    container.resize((bufferSize / sizeof(typename ContainerType::value_type)) + ((bufferSize % sizeof(typename ContainerType::value_type)) > 0 ? 1 : 0));
    char* target = reinterpret_cast<char*>(container.data());
    auto& flatheader = *reinterpret_cast<FlatHeader*>(target);
    target += sizeof(FlatHeader);
    flatheader.version = 1;
    flatheader.sizeofHeaderElement = sizeof(MCTruthHeaderElement);
    flatheader.sizeofTruthElement = sizeof(TruthElement);
    flatheader.reserved = 0;
    flatheader.nofHeaderElements = mHeaderArray.size();
    flatheader.nofTruthElements = mTruthArray.size();
    size_t copySize = flatheader.sizeofHeaderElement * flatheader.nofHeaderElements;
    memcpy(target, mHeaderArray.data(), copySize);
    target += copySize;
    copySize = flatheader.sizeofTruthElement * flatheader.nofTruthElements;
    memcpy(target, mTruthArray.data(), copySize);
    return bufferSize;
  }

  /// Resore internal vectors from a raw buffer
  /// The two vectors are resized according to the information in the \a FlatHeader
  /// struct at the beginning of the buffer. Data is copied to the vectors.
  void restore_from(const char* buffer, size_t bufferSize)
  {
    if (buffer == nullptr || bufferSize < sizeof(FlatHeader)) {
      return;
    }
    auto* source = buffer;
    auto& flatheader = *reinterpret_cast<FlatHeader const*>(source);
    source += sizeof(FlatHeader);
    if (bufferSize < sizeof(FlatHeader) + flatheader.sizeofHeaderElement * flatheader.nofHeaderElements + flatheader.sizeofTruthElement * flatheader.nofTruthElements) {
      throw std::runtime_error("inconsistent buffer size: too small");
      return;
    }
    if (flatheader.sizeofHeaderElement != sizeof(MCTruthHeaderElement) || flatheader.sizeofTruthElement != sizeof(TruthElement)) {
      // not yet handled
      throw std::runtime_error("member element sizes don't match");
    }
    // TODO: with a spectator memory ressource the vectors can be built directly
    // over the original buffer, there is the implementation for a memory ressource
    // working on a FairMQ message, here we would need two memory resources over
    // the two ranges of the input buffer
    // for now doing a copy
    mHeaderArray.resize(flatheader.nofHeaderElements);
    mTruthArray.resize(flatheader.nofTruthElements);
    size_t copySize = flatheader.sizeofHeaderElement * flatheader.nofHeaderElements;
    memcpy(mHeaderArray.data(), source, copySize);
    source += copySize;
    copySize = flatheader.sizeofTruthElement * flatheader.nofTruthElements;
    memcpy(mTruthArray.data(), source, copySize);
  }

  /// Print some info
  template <typename Stream>
  void print(Stream& stream)
  {
    stream << "MCTruthContainer index = " << getIndexedSize() << " for " << getNElements() << " elements(s), flat buffer size " << mStreamerData.size() << std::endl;
  }

  /// Inflate the object from the internal buffer
  /// The class has a specific member to store flattened data. Due to some limitations in ROOT
  /// it is more efficient to first flatten the objects to a raw buffer and empty the two vectors
  /// before serialization. This function restores the vectors from the internal raw buffer.
  /// Called from the custom streamer.
  void inflate()
  {
    if (mHeaderArray.size() > 0) {
      mStreamerData.clear();
      return;
    }
    restore_from(mStreamerData.data(), mStreamerData.size());
    mStreamerData.clear();
  }

  /// Deflate the object to the internal buffer
  /// The class has a specific member to store flattened data. Due to some limitations in ROOT
  /// it is more efficient to first flatten the objects to a raw buffer and empty the two vectors
  /// before serialization. This function stores the vectors to the internal raw buffer.
  /// Called from the custom streamer.
  void deflate()
  {
    if (mStreamerData.size() > 0) {
      clear();
      return;
    }
    mStreamerData.clear();
    flatten_to(mStreamerData);
    clear();
  }

  ClassDefNV(MCTruthContainer, 2);
}; // end class

} // namespace dataformats
} // namespace o2

#endif
