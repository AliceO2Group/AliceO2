// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MATHUTILS_VALUEMONITOR_H_
#define ALICEO2_MATHUTILS_VALUEMONITOR_H_

#include "TH1.h"
#include <unordered_map>
#include <string>

namespace o2
{
namespace utils
{

/*
 ValueMonitor: Facility to record values in a hist
    
 Mainly meant as a service class that makes it easy
 to dump variable values (within an algorithm) to a histogram
 for later visual inspection.
 The class is similar in spirit with the TreeStreamer facility
 but complementary since directly using histograms in memory.
    
 Different histograms are saved to the same file.
    
 ```C++
  ValueMonitor mon(filename);
    
  float x;
  mon.Collect<float>("x", x); --> collects x in histogram named "x"
   
  double y;
  mon.Collect<double>("y", y); --> collects y in histogram named "y"
 ```
*/
class ValueMonitor
{
 public:
  ValueMonitor(std::string filename);
  ~ValueMonitor();

  /// main interface to add a value to a histogram called "key"
  template <typename T>
  void Collect(const char* key, T value);

 private:
  std::string mFileName; // name of file where histograms are dumped to

  std::unordered_map<const char*, TH1*> mHistos; // container of histograms (identified by name)
};

namespace
{
template <typename T>
inline TH1* makeHist(const char* key)
{
  return nullptr;
}

template <>
inline TH1* makeHist<int>(const char* key)
{
  return new TH1I(key, key, 200, 0, 1);
}

template <>
inline TH1* makeHist<double>(const char* key)
{
  return new TH1D(key, key, 200, 0, 1);
}

template <>
inline TH1* makeHist<float>(const char* key)
{
  return new TH1F(key, key, 200, 0, 1);
}
} // namespace

template <typename T>
inline void ValueMonitor::Collect(const char* key, T value)
{
  // see if we have this histogram already
  TH1* h = nullptr;
  auto iter = mHistos.find(key);
  if (iter == mHistos.end()) {
    auto newHist = makeHist<T>(key);
    newHist->SetCanExtend(TH1::kAllAxes);
    mHistos[key] = newHist;
    h = newHist;
  } else {
    h = (*iter).second;
  }
  h->Fill(value);
}

} // namespace utils
} // namespace o2

#endif
