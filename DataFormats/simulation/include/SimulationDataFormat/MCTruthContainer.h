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
}
}

#endif
