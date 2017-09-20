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

namespace o2
{
namespace dataformats
{
// a simple struct having information about truth elements for particular indices:
// how many associations we have and where they start in the storage
struct MCTruthHeaderElement {
  MCTruthHeaderElement() = default; // for ROOT IO

  MCTruthHeaderElement(ushort s, uint i) : size(s), index(i) {}
  ushort size = 0; // the number of entries
  uint index = 0;  // the index into the actual MC track storage
  ClassDefNV(MCTruthHeaderElement, 1);
};

// a container to hold and manage MC truth information
// the actual MCtruth type is a generic template type and can be supplied by the user
// It is meant to manage associations from one "dataobject" identified by an index into an array
// to multiple TruthElements

// note that we inherit from TObject just to be able to register this thing with the FairRootManager
// (which might not be necessary in the future)
template <typename TruthElement>
class MCTruthContainer : public TNamed
{
 private:
  std::vector<MCTruthHeaderElement>
    mHeaderArray;                        // the header structure array serves as an index into the actual storage
  std::vector<TruthElement> mTruthArray; // the buffer containing the actual truth information

 public:
  // constructor
  MCTruthContainer() = default;

  ~MCTruthContainer() final = default;

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
  gsl::span<TruthElement> getLabels(int dataindex) {
    if(dataindex >= getIndexedSize()) return gsl::span<TruthElement>();
    return gsl::span<TruthElement>(&mTruthArray[mHeaderArray[dataindex].index], mHeaderArray[dataindex].size);
  }

  // get individual const "view" container for a given data index
  // the caller can't do modifications on this view
  gsl::span<const TruthElement> getLabels(int dataindex) const {
    if(dataindex >= getIndexedSize()) return gsl::span<const TruthElement>();
    return gsl::span<const TruthElement>(&mTruthArray[mHeaderArray[dataindex].index], mHeaderArray[dataindex].size);
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
      assert(dataindex == mHeaderArray.size());
      // add a new one
      mHeaderArray.emplace_back(0, mTruthArray.size());
    }
    auto& header = mHeaderArray[dataindex];
    header.size++;
    mTruthArray.emplace_back(element);
  }

  ClassDefOverride(MCTruthContainer, 1);
}; // end class

// -------------------- similar structure resembling more a list ---------------------
 
// a simple struct having information about truth elements for particular indices:
// how many associations we have and where they start in the storage
struct MCTruthLinkableHeaderElement {
  MCTruthLinkableHeaderElement() = default; // for ROOT IO

  MCTruthLinkableHeaderElement(uint i, int l, ushort s) : index(i), lastindex(l), size(s) {}
  uint index = 0; // the index into the actual MC track storage
  uint lastindex = 0; // pointer to current last element
  ushort size = 0; // the number of entries
  ClassDefNV(MCTruthLinkableHeaderElement, 1);
};

#include <iterator>  // std::iterator, std::input_iterator_tag

// a container to hold and manage MC truth information
// the actual MCtruth type is a generic template type and can be supplied by the user
// It is meant to manage associations from one "dataobject" identified by an index into an array
// to multiple TruthElements

// note that we inherit from TObject just to be able to register this thing with the FairRootManager
// (which might not be necessary in the future)
template <typename TruthElement>
class MCTruthContainerList : public TNamed
{
 private:
  std::vector<MCTruthLinkableHeaderElement>
    mHeaderArray;                        // the header structure array serves as an index into the actual storage
  using StoredLabelType = std::pair<TruthElement, int>; // the element as well as index/location of next label
  std::vector<StoredLabelType> mTruthArray; // the buffer containing the actual truth information

  // an iterator class to iterate over truthelements 
  class Iterator : public std::iterator<std::input_iterator_tag, TruthElement>
  {
    private:
      std::vector<StoredLabelType> &mLabelsRef; // reference to labels vector
      int index; // startindex
    public:
      Iterator(std::vector<StoredLabelType> &v, int i) : mLabelsRef(v), index(i) {}
      Iterator(const Iterator& it) : mLabelsRef(it.mLabelsRef), index(it.index) {}
      Iterator& operator=(const Iterator& it) {
        mLabelsRef = it.mLabelsRef;
	index = it.index;
        return *this;
      }
      
      // go to the next element as indicated by second entry of StoredLabelType
      Iterator& operator++() {index=mLabelsRef[index].second;return *this;}

      // pointing to same storage??
      bool operator==(const Iterator& rhs) const {return index==rhs.index;}
      bool operator!=(const Iterator& rhs) const {return index!=rhs.index;}
      TruthElement& operator*() {return mLabelsRef[index].first;}
  };

  // a proxy class offering a (non-owning) container view on labels of a certain data index
  // container offers basic forward iterator functionality
  class LabelView {
   private:
    int dataindex;
    std::vector<MCTruthLinkableHeaderElement>& mHeaderArrayRef;
    std::vector<StoredLabelType>& mTruthArrayRef;
    
    
   public:
    constexpr LabelView(int i, std::vector<MCTruthLinkableHeaderElement> &v1, std::vector<StoredLabelType> &v2) : dataindex(i), mHeaderArrayRef(v1), mTruthArrayRef(v2) {}
    // iterators to access and loop over the elements
    Iterator begin() { return Iterator(mTruthArrayRef, mHeaderArrayRef[dataindex].index); }
    Iterator end() { return Iterator(mTruthArrayRef, -1); }

    // get number of labels
    size_t size() const {return mHeaderArrayRef[dataindex].size;}
  };
  
 public:
  // constructor
  MCTruthContainerList() = default;
  ~MCTruthContainerList() final = default;

  // access
  MCTruthLinkableHeaderElement getMCTruthHeader(uint dataindex) const { return mHeaderArray[dataindex]; }
  // access the element directly (can be encapsulated better away)... needs proper element index
  // which can be obtained from the MCTruthHeader startposition and size
  TruthElement const& getElement(uint elementindex) const { return mTruthArray[elementindex].first; }

  // return the number of original data indexed here
  size_t getIndexedSize() const { return mHeaderArray.size(); }
  // return the number of elements managed in this container
  size_t getNElements() const { return mTruthArray.size(); }

  // return an iterator over labels for this container
  // try it out just for zero element
  Iterator begin(int dataindex) { return Iterator(mTruthArray, mHeaderArray[dataindex].index); }
  Iterator end() { return Iterator(mTruthArray, -1); }

  LabelView getLabels(int dataindex) { return LabelView(dataindex, mHeaderArray, mTruthArray); }
  
  void clear()
  {
    mHeaderArray.clear();
    mTruthArray.clear();
  }

  // add element for a particular dataindex
  // this version supports random insertion naturally
  void addElement(uint dataindex, TruthElement const& element)
  {
    mTruthArray.emplace_back(std::make_pair(element,-1));
    // something exists for this dataindex already
    if (dataindex < mHeaderArray.size()) {
      auto& header = mHeaderArray[dataindex];
      // increase size
      header.size++;
      const auto lastindex = mTruthArray.size();
      // fix link at previous last index
      mTruthArray[header.lastindex].second = lastindex;
      // new last index
      header.lastindex = lastindex;
    }
    else {
      // we support only appending in order?
      assert(dataindex == mHeaderArray.size());
      // add a new header element; pointing to last slot in mTruthArray
      mHeaderArray.emplace_back(mTruthArray.size()-1, mTruthArray.size()-1, 1);
    }
  }

  ClassDefOverride(MCTruthContainerList, 1);
}; // end class



}
}

#endif
