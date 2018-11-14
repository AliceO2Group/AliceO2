// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_Metric2DViewInfo_H_INCLUDED
#define o2_framework_Metric2DViewInfo_H_INCLUDED

#include <functional>
#include <cstddef>
#include <string>
#include <vector>

namespace o2
{
namespace framework
{

struct MetricInfo;

/// This allows keeping track of the metrics which should be grouped together
/// in some sort of 2D representation.
struct Metric2DViewIndex {
  using Updater = std::function<void(std::string const&, MetricInfo const&, int)>;
  /// The prefix in the metrics store to be used for the view
  std::string prefix;
  /// The size in X of the metrics
  int w = 0;
  /// The size in Y of the metrics
  int h = 0;
  /// The row major list of indices for the metrics which compose the 2D view
  std::vector<std::size_t> indexes = {};
  /// Whether or not the view is ready to be used.
  bool isComplete() const { return (w * h) != 0; }

  static Updater getUpdater(Metric2DViewIndex& view);
};

} // namespace framework
} // namespace o2

#endif // o2_framework_Metric2DViewInfo_H_INCLUDED
