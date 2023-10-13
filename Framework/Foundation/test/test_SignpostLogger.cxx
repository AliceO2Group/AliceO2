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

// We need the implementation on Apple, to get the logger based signposts
// We also need the implementation in release mode, because the logger based signposts are not available in the release build
#if defined(__APPLE__) || defined(NDEBUG)
#define O2_SIGNPOST_IMPLEMENTATION
#endif
#define O2_FORCE_LOGGER_SIGNPOST 1
#include "Framework/Signpost.h"
#include <iostream>

O2_DECLARE_LOG(test_Signpost2, "my category2");

int main(int argc, char** argv)
{
  O2_DECLARE_LOG(test_Signpost, "my category");
  O2_DECLARE_DYNAMIC_LOG(test_SignpostDynamic);

  std::cout << "Loggers: " << std::endl;
  o2_walk_logs([](char const* name, void* log, void*) -> bool {
    std::cout << "  - name: " << name << " " << log << std::endl;
    return true;
  },
               nullptr);

  O2_LOG_DEBUG(test_Signpost, "%s %d", "test_Signpost", 1);
  O2_SIGNPOST_ID_GENERATE(id, test_Signpost);
  O2_SIGNPOST_ID_GENERATE(id2, test_Signpost);
  O2_SIGNPOST_START(test_Signpost, id, "Test category", "This is a test signpost");
  O2_SIGNPOST_START(test_Signpost, id2, "Test category", "A sepaarate interval");
  O2_SIGNPOST_EVENT_EMIT(test_Signpost, id, "Test category", "An event in an interval");
  O2_SIGNPOST_END(test_Signpost, id, "Test category", "End of the first interval");
  O2_SIGNPOST_END(test_Signpost, id2, "Test category", "A sepaarate interval");
  O2_SIGNPOST_ID_FROM_POINTER(id3, test_Signpost, &id2);
  O2_SIGNPOST_START(test_Signpost, id3, "Test category", "A signpost interval from a pointer");
  O2_SIGNPOST_END(test_Signpost, id3, "Test category", "A signpost interval from a pointer");

  // This has an engineering type, which we will not use on Linux / FairLogger
  O2_SIGNPOST_ID_FROM_POINTER(id4, test_Signpost, &id3);
  O2_SIGNPOST_START(test_Signpost, id4, "Test category", "A signpost with an engineering type formatter " O2_ENG_TYPE(size - in - bytes, "d"), 1);
  O2_SIGNPOST_END(test_Signpost, id4, "Test category", "A signpost interval from a pointer");

  O2_SIGNPOST_START(test_SignpostDynamic, id, "Test category", "This is dynamic signpost which you will not see, because they are off by default");
  O2_SIGNPOST_END(test_SignpostDynamic, id, "Test category", "This is dynamic signpost which you will not see, because they are off by default");
  O2_LOG_ENABLE_DYNAMIC(test_SignpostDynamic);
#ifdef __APPLE__
  // On Apple there is no way to turn on signposts in the logger, so we do not display this message
  O2_SIGNPOST_START(test_SignpostDynamic, id, "Test category", "This is dynamic signpost which you will see, because we turned them on");
  O2_SIGNPOST_END(test_SignpostDynamic, id, "Test category", "This is dynamic signpost which you will see, because we turned them on");
#endif
}
