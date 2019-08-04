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

#include <TNamed.h>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl> // for guideline support library; array_view
#include <type_traits>

namespace o2
{
namespace dataformats
{
// a simple struct having information about truth elements for particular indices:
// how many associations we have and where they start in the storage
struct MCTruthHeaderElement {
  MCTruthHeaderElement() = default; // for ROOT IO

  MCTruthHeaderElement(uint i) : index(i) {}
  uint index = -1; // the index into the actual MC track storage (-1 if invalid)
  ClassDefNV(MCTruthHeaderElement, 1);
};

// A container to hold and manage MC truth information/labels.
// The actual MCtruth type is a generic template type and can be supplied by the user
// It is meant to manage associations from one "dataobject" identified by an index into an array
// to multiple TruthElements

template <typename TruthElement>
class MCTruthContainer
{
 private:
  std::vector<MCTruthHeaderElement>
    mHeaderArray;                        // the header structure array serves as an index into the actual storage
  std::vector<TruthElement> mTruthArray; // the buffer containing the actual truth information

  size_t getSize(uint dataindex) const
  {
    // calculate size / number of labels from a difference in pointed indices
    const auto size = (dataindex < mHeaderArray.size() - 1)
                        ? mHeaderArray[dataindex + 1].index - mHeaderArray[dataindex].index
                        : mTruthArray.size() - mHeaderArray[dataindex].index;
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

  // access
  MCTruthHeaderElement getMCTruthHeader(uint dataindex) const { return mHeaderArray[dataindex]; }
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
    return gsl::span<TruthElement>(&mTruthArray[mHeaderArray[dataindex].index], getSize(dataindex));
  }

  // get individual const "view" container for a given data index
  // the caller can't do modifications on this view
  gsl::span<const TruthElement> getLabels(uint dataindex) const
  {
    if (dataindex >= getIndexedSize()) {
      return gsl::span<const TruthElement>();
    }
    return gsl::span<const TruthElement>(&mTruthArray[mHeaderArray[dataindex].index], getSize(dataindex));
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

  ClassDefNV(MCTruthContainer, 1);
}; // end class

} // namespace dataformats
} // namespace o2

#endif
