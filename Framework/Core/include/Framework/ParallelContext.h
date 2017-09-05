// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_PARALLELCONTEXT_H
#define FRAMEWORK_PARALLELCONTEXT_H

namespace o2 {
namespace framework {

class ParallelContext {
public:
  // FIXME: find better names... rank1D and rank1DSize?
  ParallelContext(size_t index1D, size_t index1DSize)
    : mIndex1D{index1D},
      mIndex1DSize{index1DSize}
  {
  }

  size_t index1D() const { return mIndex1D; }
  size_t index1DSize() const {return mIndex1DSize; };
private:
  size_t mIndex1D;
  size_t mIndex1DSize;
};

}
}
#endif
