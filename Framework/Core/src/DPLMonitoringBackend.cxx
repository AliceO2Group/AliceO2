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

#include "DPLMonitoringBackend.h"
#include "Framework/DriverClient.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RuntimeError.h"
#include <fmt/format.h>
#include <sstream>

namespace o2::framework
{

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;


DPLMonitoringBackend::DPLMonitoringBackend(ServiceRegistryRef registry)
  : mRegistry{registry}
{
}

void DPLMonitoringBackend::addGlobalTag(std::string_view name, std::string_view value)
{
  // FIXME: tags are ignored by DPL in any case...
  mTagString += fmt::format("{}{}={}", mTagString.empty() ? "" : ",", name.data(), value);
}

void DPLMonitoringBackend::send(std::vector<o2::monitoring::Metric>&& metrics)
{
  for (auto& m : metrics) {
    send(m);
  }
}

inline unsigned long convertTimestamp(const std::chrono::time_point<std::chrono::system_clock>& timestamp)
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           timestamp.time_since_epoch())
    .count();
}

void DPLMonitoringBackend::send(o2::monitoring::Metric const& metric)
{
  std::array<char, 4096> buffer;
  auto mStream = fmt::format_to(buffer.begin(), "[METRIC] {}", metric.getName());
  for (auto& value : metric.getValues()) {
    auto stringValue = std::visit(overloaded{
                                    [](const std::string& value) -> std::string { return value; },
                                    [](auto value) -> std::string { return std::to_string(value); }},
                                  value.second);
    if (metric.getValuesSize() == 1) {
      mStream = fmt::format_to(mStream, ",{} {}", metric.getFirstValueType(), stringValue);
    } else {
      mStream = fmt::format_to(mStream, " {}={}", value.first, stringValue);
    }
  }
  // FIXME: tags are ignored by the DPL backend in any case...
  mStream = fmt::format_to(mStream, " {} {}", convertTimestamp(metric.getTimestamp()), mTagString);

  bool first = mTagString.empty();
  for (const auto& [key, value] : metric.getTags()) {
    if (!first) {
      mStream = fmt::format_to(mStream, ",");
    }
    first = true;
    mStream = fmt::format_to(mStream, "{}={}", o2::monitoring::tags::TAG_KEY[key], o2::monitoring::tags::GetValue(value));
  }
  mStream = fmt::format_to(mStream, "\n");
  auto size = std::distance(buffer.begin(), mStream);
  if (size + 1 >= 4096) {
    throw runtime_error_f("Metric too long");
  }
  buffer[size] = '\0';
  mRegistry.get<framework::DriverClient>().tell(buffer.data(), size);
}

} // namespace o2::framework
