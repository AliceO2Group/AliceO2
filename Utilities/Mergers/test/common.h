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

#ifndef ALICEO2_MERGER_TEST_COMMON_H_
#define ALICEO2_MERGER_TEST_COMMON_H_

#include <sstream>
#include <gsl/span>
#include <TH1.h>
#include <Framework/CallbackService.h>

namespace o2::mergers::test
{
inline auto to_span(const TH1F& histo)
{
  return gsl::span(histo.GetArray(), histo.GetSize());
}

void registerCallbacksForTestFailure(framework::CallbackService& cb, std::shared_ptr<bool> success)
{
  cb.set<framework::CallbackService::Id::EndOfStream>([success](framework::EndOfStreamContext& ctx) {
    if (*success == false) {
      LOG(fatal) << "Received an EndOfStream without having received the expected object";
    }
  });
  cb.set<framework::CallbackService::Id::Stop>([success]() {
    if (*success == false) {
      LOG(fatal) << "STOP transition without having received the expected object";
    }
  });
  cb.set<framework::CallbackService::Id::ExitRequested>([success](framework::ServiceRegistryRef) {
    if (*success == false) {
      LOG(fatal) << "EXIT transition without having received the expected object";
    }
  });
}

} // namespace o2::mergers::test

namespace std
{

template <typename T, size_t Size>
inline std::string to_string(gsl::span<T, Size> span)
{
  std::stringstream ss{};
  for (size_t i = 0; auto v : span) {
    ss << v;
    if (++i != span.size()) {
      ss << " ";
    }
  }
  return std::move(ss).str();
}

template <size_t Size>
inline std::string to_string(const std::array<float, Size>& arr)
{
  return to_string(gsl::span(arr));
}

inline std::string to_string(const TH1F& histo)
{
  return to_string(o2::mergers::test::to_span(histo));
}

template <typename Deleter>
inline std::string to_string(const std::unique_ptr<const TH1F, Deleter>& histo_ptr)
{
  return to_string(o2::mergers::test::to_span(*histo_ptr.get()));
}

template <size_t Size>
inline std::string to_string(const std::vector<std::array<float, Size>>& arrays)
{
  std::string res{};
  for (size_t i = 0; const auto& array : arrays) {
    res.append("[");
    res.append(to_string(gsl::span(array)));
    res.append("]");
  }
  return res;
}

template <typename Deleter>
inline std::string to_string(const std::unique_ptr<const std::vector<TObject*>, Deleter>& histos)
{
  std::string res{};
  for (size_t i = 0; const auto& histo : *histos) {
    res.append("[");
    res.append(to_string(*dynamic_cast<TH1F const*>(histo)));
    res.append("]");
  }
  return res;
}

} // namespace std

#endif
