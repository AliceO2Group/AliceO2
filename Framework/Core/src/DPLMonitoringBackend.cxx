// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DPLMonitoringBackend.h"
#include "Framework/DriverClient.h"
#include "Framework/ServiceRegistry.h"
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

inline unsigned long DPLMonitoringBackend::convertTimestamp(const std::chrono::time_point<std::chrono::system_clock>& timestamp)
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           timestamp.time_since_epoch())
    .count();
}

DPLMonitoringBackend::DPLMonitoringBackend(ServiceRegistry& registry)
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

void DPLMonitoringBackend::send(o2::monitoring::Metric const& metric)
{
  std::ostringstream mStream;
  mStream << "[METRIC] " << metric.getName();
  for (auto& value : metric.getValues()) {
    auto stringValue = std::visit(overloaded{
                                    [](const std::string& value) -> std::string { return value; },
                                    [](auto value) -> std::string { return std::to_string(value); }},
                                  value.second);
    if (metric.getValuesSize() == 1) {
      mStream << ',' << metric.getFirstValueType() << ' ' << stringValue;
    } else {
      mStream << ' ' << value.first << '=' << stringValue;
    }
  }
  // FIXME: tags are ignored by the DPL backend in any case...
  mStream << ' ' << convertTimestamp(metric.getTimestamp()) << ' ' << mTagString;

  bool first = mTagString.empty();
  for (const auto& [key, value] : metric.getTags()) {
    if (!first) {
      mStream << ',';
    }
    first = true;
    mStream << o2::monitoring::tags::TAG_KEY[key] << "=" << o2::monitoring::tags::GetValue(value);
  }
  mStream << '\n';
  mRegistry.get<framework::DriverClient>().tell(mStream.str().c_str());
}

} // namespace o2::framework
