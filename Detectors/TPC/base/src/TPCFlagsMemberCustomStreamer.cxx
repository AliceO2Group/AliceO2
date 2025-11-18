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
    std::vector<int> R__stl;
    R__stl.clear();
    int R__n;
    R__b >> R__n;
    R__stl.reserve(R__n);
    for (int R__i = 0; R__i < R__n; R__i++) {
      Int_t readtemp;
      R__b >> readtemp;
      R__stl.push_back(readtemp);
    }
    auto data = reinterpret_cast<unsigned short*>(R__stl.data());
    for (int i = 0; i < R__n; ++i) {
      obj->push_back(static_cast<o2::tpc::PadFlags>(data[i]));
    }
  } else {
    // We always save things with the old format.
    R__b << (int)obj->size() / 2;
    for (size_t i = 0; i < obj->size(); i++) {
      R__b << (short)obj->at(i);
    }
  }
}

// register the streamer via static global initialization (on library load)
namespace ROOT
{
static __attribute__((used)) int _R__dummyStreamer_3 =
  ([]() {
    auto cl = TClass::GetClass<o2::tpc::CalArray<o2::tpc::PadFlags>>();
    if (cl) {
      cl->AdoptMemberStreamer("mData", new TMemberStreamer(MemberVectorPadFlagsStreamer));
    } else {
      // we should never come here ... and if we do we should assert/fail
      assert(false);
    }
    return 0;
  })();
} // namespace ROOT
