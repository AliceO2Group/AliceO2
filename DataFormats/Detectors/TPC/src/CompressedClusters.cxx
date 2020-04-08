// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CompressedClusters.cxx
/// \brief Container to store compressed TPC cluster data

#include "DataFormatsTPC/CompressedClusters.h"
#include "DataFormatsTPC/CompressedClustersHelpers.h"
#include "TBuffer.h"
#include <vector>
#include <gsl/span>

namespace o2::tpc
{

template <>
void CompressedClusters::Streamer(TBuffer& R__b)
{
  // the custom streamer for CompressedClusters
  if (R__b.IsReading()) {
    R__b.ReadClassBuffer(CompressedClusters::Class(), this);
    gsl::span flatdata{this->flatdata, this->flatdataSize};
    CompressedClustersHelpers::restoreFrom(flatdata, *this);
  } else {
    std::vector<char> flatdata;
    // member flatdata is nullptr unless it is an already extracted object which
    // is streamed again. Overriding the existing flatbuffer is a potential
    // memory leak
    // Note: lots of cornercases here which are diffucult to catch with the design
    // of CompClusters container (which is optimized to be used with GPU). Main concern
    // is if an extracted object is not read-only and the counters are changed.
    // TODO: we can add a consistency check whether the size of the flatbuffer
    // and the pointers match the object.
    bool isflat = this->flatdata != nullptr;
    if (!isflat) {
      CompressedClustersHelpers::flattenTo(flatdata, *this);
      this->flatdataSize = flatdata.size();
      this->flatdata = flatdata.data();
    }
    R__b.WriteClassBuffer(CompressedClusters::Class(), this);
    if (!isflat) {
      this->flatdataSize = 0;
      this->flatdata = nullptr;
    }
  }
}

} // namespace o2::tpc
