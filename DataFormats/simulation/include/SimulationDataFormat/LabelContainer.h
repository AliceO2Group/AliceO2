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

/// \brief Definition of a container to keep/associate and arbitrary
/// number of labels associated to an index with contiguous or non-contiguous label storage
// \author Sandro Wenzel - June 2017

#ifndef ALICEO2_DATAFORMATS_LABELCONTAINER_H_
#define ALICEO2_DATAFORMATS_LABELCONTAINER_H_

#include <TNamed.h>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <iterator>
#include <gsl/gsl> // for guideline support library; array_view

namespace o2
{
namespace dataformats
{

// a label container with both contiguous and non-contiguous modes
// for label storage
// NOTE: the current specialization for contiguous storage is not
// yet on par with MCTruthContainer ... but eventually we might unify
// everything and keep only the more generic and configurable version
template <typename LabelType, bool isContiguousStorage = true>
class LabelContainer
{
 public:
  struct HeaderElementContinuous {
    HeaderElementContinuous() = default; // for ROOT IO
    HeaderElementContinuous(unsigned short s, unsigned int i) : size(s), index(i) {}
    unsigned int index = 0;  // index of first label in the actual label storage
    unsigned short size = 0; // total number of labels
    ClassDefNV(HeaderElementContinuous, 1);
  };

  struct HeaderElementLinked {
    HeaderElementLinked() = default; // for ROOT IO
    HeaderElementLinked(unsigned int i, int l, unsigned short s) : index(i), lastindex(l), size(s) {}
    unsigned int index = 0;     // index of first label in the actual label storage
    unsigned int lastindex = 0; // index of last label in the actual label storage
    unsigned short size = 0;    // total number of labels
    ClassDefNV(HeaderElementLinked, 1);
  };

  using HeaderElement = typename std::conditional<isContiguousStorage, HeaderElementContinuous, HeaderElementLinked>::type;
  using StoredLabelType = typename std::conditional<isContiguousStorage, LabelType, std::pair<LabelType, int>>::type;

  // internal functions allowing the iterator implementation to be completely generic
  static unsigned int getNextIndex(unsigned int index, std::vector<LabelType> const& /*labels*/)
  {
    return index + 1;
  }
  static unsigned int getNextIndex(unsigned int index, std::vector<std::pair<LabelType, int>> const& labels)
  {
    return labels[index].second;
  }
  static LabelType& dereference(std::vector<LabelType>& v, int index)
  {
    return v[index];
  }
  static LabelType& dereference(std::vector<std::pair<LabelType, int>>& v, int index)
  {
    return v[index].first;
  }
  static int lastIndex(HeaderElementContinuous const& h)
  {
    return h.index + h.size;
  }
  static int lastIndex(HeaderElementLinked const& h)
  {
    // -1 since this is indication of end of linked list
    return -1;
  }

  // an iterator class to iterate over truthelements
  class Iterator : public std::iterator<std::input_iterator_tag, LabelType>
  {
   private:
    std::vector<StoredLabelType>& mLabelsRef; // reference to labels vector
    int index;                                // startindex
   public:
    Iterator(std::vector<StoredLabelType>& v, int i) : mLabelsRef(v), index(i) {}
    Iterator(const Iterator& it) : mLabelsRef(it.mLabelsRef), index(it.index) {}
    Iterator& operator=(const Iterator& it)
    {
      mLabelsRef = it.mLabelsRef;
      index = it.index;
      return *this;
    }

    // go to the next element as indicated by second entry of StoredLabelType
    Iterator& operator++()
    {
      index = getNextIndex(index, mLabelsRef);
      return *this;
    }

    bool operator==(const Iterator& rhs) const { return index == rhs.index; }
    bool operator!=(const Iterator& rhs) const { return index != rhs.index; }
    LabelType& operator*() { return dereference(mLabelsRef, index); }
  };

  // a proxy class offering a (non-owning) container view on labels of a given data index
  // container offers basic forward iterator functionality
  class LabelView
  {
   private:
    int dataindex;
    std::vector<HeaderElement>& mHeaderArrayRef;
    std::vector<StoredLabelType>& mLabelArrayRef;

   public:
    constexpr LabelView(int i, std::vector<HeaderElement>& v1, std::vector<StoredLabelType>& v2) : dataindex(i), mHeaderArrayRef(v1), mLabelArrayRef(v2) {}

    // begin + end iterators to loop over the labels
    Iterator begin()
    {
      return dataindex < mHeaderArrayRef.size() ? Iterator(mLabelArrayRef, mHeaderArrayRef[dataindex].index) : Iterator(mLabelArrayRef, 0);
    }

    Iterator end()
    {
      return dataindex < mHeaderArrayRef.size() ? Iterator(mLabelArrayRef, lastIndex(mHeaderArrayRef[dataindex])) : Iterator(mLabelArrayRef, 0);
    }

    // get number of labels
    size_t size() const { return dataindex < mHeaderArrayRef.size() ? mHeaderArrayRef[dataindex].size : 0; }
  };

  static void addLabelImpl(int dataindex, std::vector<HeaderElementContinuous>& headerv, std::vector<LabelType>& labelv, LabelType const& label)
  {
    if (dataindex < headerv.size()) {
      // look if we have something for this dataindex already
      // must currently be the last one
      if (dataindex != (headerv.size() - 1)) {
        throw std::runtime_error("LabelContainer: unsupported code path");
      }
    } else {
      assert(dataindex == headerv.size());
      // add a new one
      headerv.emplace_back(0, labelv.size());
    }
    auto& header = headerv[dataindex];
    header.size++;
    labelv.emplace_back(label);
  }

  static void addLabelImpl(int dataindex, std::vector<HeaderElementLinked>& headerv, std::vector<std::pair<LabelType, int>>& labelv, LabelType const& label)
  {
    labelv.emplace_back(std::make_pair(label, -1));
    // something exists for this dataindex already
    if (dataindex < headerv.size()) {
      auto& header = headerv[dataindex];
      // increase size
      header.size++;
      const auto lastindex = labelv.size() - 1;
      // fix link at previous last index
      labelv[header.lastindex].second = lastindex;
      // new last index
      header.lastindex = lastindex;
    } else {
      // we support only appending in order?
      assert(dataindex == headerv.size());
      // add a new header element; pointing to last slot in mTruthArray
      const auto lastpos = labelv.size() - 1;
      headerv.emplace_back(lastpos, lastpos, 1);
    }
  }

  // declaring the data members
  std::vector<HeaderElement> mHeaderArray;  // the header structure array serves as an index into the actual storage
  std::vector<StoredLabelType> mLabelArray; // the buffer containing the actual truth information

 public:
  // constructor
  LabelContainer() = default;
  ~LabelContainer() = default;

  // reserve some initial space in the actual storage
  // (to hold at least n labels)
  void reserve(int n)
  {
    mHeaderArray.reserve(n);
    mLabelArray.reserve(n);
  }

  void clear()
  {
    mHeaderArray.clear();
    mLabelArray.clear();
  }

  /// add a label for a dataindex
  void addLabel(unsigned int dataindex, LabelType const& label)
  {
    // refer to concrete specialized implementation
    addLabelImpl(dataindex, mHeaderArray, mLabelArray, label);
  }

  /// get a container view on labels allowing use standard forward iteration in user code
  LabelView getLabels(int dataindex) { return LabelView(dataindex, mHeaderArray, mLabelArray); }

  /// fill an external vector container with labels
  /// might be useful to perform additional operations such as sorting on the labels;
  /// the external vector can be reused to avoid allocations/deallocs)
  void fillVectorOfLabels(int dataindex, std::vector<LabelType>& v)
  {
    /// fixme: provide a template specialized fast version for contiguous storage
    v.clear();
    for (auto& e : getLabels(dataindex)) {
      v.push_back(e);
    }
  }
}; // end class

} // namespace dataformats
} // namespace o2

#endif
