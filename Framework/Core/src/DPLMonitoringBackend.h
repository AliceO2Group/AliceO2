// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DPLMONITORINGBACKEND_H_
#define O2_FRAMEWORK_DPLMONITORINGBACKEND_H_

#include "Monitoring/Backend.h"
#include <string>

namespace o2::framework
{

struct ServiceRegistry;

/// \brief Prints metrics to standard output via std::cout
class DPLMonitoringBackend final : public o2::monitoring::Backend
{
 public:
  /// Default constructor
  DPLMonitoringBackend(ServiceRegistry& registry);

  /// Default destructor
  ~DPLMonitoringBackend() override = default;

  /// Prints metric
  /// \param metric           reference to metric object
  void send(const o2::monitoring::Metric& metric) override;

  /// Prints vector of metrics
  /// \@param metrics  vector of metrics
  void send(std::vector<o2::monitoring::Metric>&& metrics) override;

  /// Adds tag
  /// \param name         tag name
  /// \param value        tag value
  void addGlobalTag(std::string_view name, std::string_view value) override;

 private:
  /// Converts timestamp to unsigned long (miliseconds from epoch)
  /// \param timestamp    timestamp in std::chrono::time_point format
  /// \return             timestamp as unsigned long (miliseconds from epoch)
  unsigned long convertTimestamp(const std::chrono::time_point<std::chrono::system_clock>& timestamp);

  std::string mTagString;    ///< Global tagset (common for each metric)
  const std::string mPrefix; ///< Metric prefix
  ServiceRegistry& mRegistry;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DPLMONITORINGBACKEND_H_
