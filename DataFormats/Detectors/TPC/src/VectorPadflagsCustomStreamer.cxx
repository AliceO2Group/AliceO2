// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

using std::vector;

void VectorPadFlagsStreamer(TBuffer& R__b, void* objp)
{
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

#define RootStreamerLocal(name, STREAMER)     \
  namespace ROOT                              \
  {                                           \
                                              \
  /** \cond HIDDEN_SYMBOLS */                 \
  static auto _R__UNIQUE_(R__dummyStreamer) = \
    []() { TClass::GetClass<name>()->SetStreamerFunc(STREAMER); return 0; }();                               \
  /** \endcond */                             \
  R__UseDummy(_R__UNIQUE_(R__dummyStreamer)); \
  }

// Let's not try to fix the old ROOT version, so that we can build
// the new ROOT with the patched code in the CI.
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 33, 00)
RootStreamerLocal(vector<o2::tpc::PadFlags>, VectorPadFlagsStreamer);
#endif
