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

#include "TPCBase/CalArray.h"
#include <TMemberStreamer.h>
#include <TBuffer.h>
#include <TROOT.h>
#include <DataFormatsTPC/Defs.h>
#include <iostream>

// to enable assert statements
#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#endif

// The following code provides a specific ROOT I/O streaming method
// for the mData member of CalArray<o2::tpc::PadFlags>
// This member was written incorrectly to the TFile in ROOT versions < 6.36, causing
// segfaults when reading on ARM64 (occassionally).
// We continue to write it in the incorrect format and fix the reading back.
//
// Two layouts exist on disk, both with the count N/2 (N = number of flags):
// - written by ROOT < 6.33 (no custom streamer): N/2 ints, each holding a pair of
//   flags as they sit in memory;
// - written by this streamer (ROOT >= 6.33): N single shorts.
// Reading the second layout as ints swaps every pair of adjacent flags, so the
// layout is chosen by the ROOT version that wrote the buffer. The class version
// did not change, so this is the only marker.

// See also:
// - https://github.com/root-project/root/pull/17009
// - https://its.cern.ch/jira/browse/O2-4671

void MemberVectorPadFlagsStreamer(TBuffer& R__b, void* objp, int n)
{
  if (n != 1) {
    std::cerr << "Error in MemberVectorPadFlagsStreamer : Unexpected n " << n << std::endl;
    return;
  }
  std::vector<o2::tpc::PadFlags>* obj = static_cast<std::vector<o2::tpc::PadFlags>*>(objp);
  if (R__b.IsReading()) {
    obj->clear();
    int R__n;
    R__b >> R__n;
    // ROOT version of the file the buffer belongs to (e.g. 63206); 0 without a file (e.g. TMessage):
    // then writer and reader run the same ROOT
    int writerVersion = R__b.GetVersionOwner();
    if (writerVersion <= 0) {
      writerVersion = gROOT->GetVersionInt();
    }
    constexpr size_t bloatfactor = sizeof(int) / sizeof(o2::tpc::PadFlags);
    if (writerVersion >= 63300) {
      // written by this streamer: N single shorts
      obj->reserve(bloatfactor * R__n);
      for (size_t i = 0; i < bloatfactor * R__n; ++i) {
        Short_t readtemp;
        R__b >> readtemp;
        obj->push_back(static_cast<o2::tpc::PadFlags>(static_cast<unsigned short>(readtemp)));
      }
      return;
    }
    // written by ROOT < 6.33: N/2 ints, each a pair of flags in memory order
    std::vector<int> R__stl;
    R__stl.reserve(R__n);
    for (int R__i = 0; R__i < R__n; R__i++) {
      Int_t readtemp;
      R__b >> readtemp;
      R__stl.push_back(readtemp);
    }
    auto data = reinterpret_cast<unsigned short*>(R__stl.data());
    for (size_t i = 0; i < bloatfactor * R__n; ++i) {
      obj->push_back(static_cast<o2::tpc::PadFlags>(data[i]));
    }
  } else {
    // count N/2, then N single shorts: NOT the old (ROOT < 6.33) layout of N/2 ints; the reader above
    // tells the two apart by the ROOT version. Unchanged, so that existing objects stay readable.
    R__b << (int)obj->size() / 2;
    for (size_t i = 0; i < obj->size(); i++) {
      R__b << (short)obj->at(i);
    }
  }
}

// register the streamer via static global initialization (on library load)
// the streamer is only correct in combination with new ROOT
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 33, 00)
namespace ROOT
{
static __attribute__((used)) int _R__dummyStreamer_3 =
  ([]() {
    if (!getenv("TPC_PADFLAGS_STREAMER_OFF")) {
      ROOT::GenerateInitInstance((o2::tpc::CalArray<o2::tpc::PadFlags> *)nullptr)->AdoptMemberStreamer("mData", new TMemberStreamer(MemberVectorPadFlagsStreamer));
    }
    return 0;
  })();
} // namespace ROOT
#endif
