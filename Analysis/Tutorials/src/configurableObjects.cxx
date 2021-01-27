// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/configurableCut.h"

#include <sstream>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

/// This task demonstrates how to use configurable to wrap classes
/// use it with supplied configuration: "configurableObject.json"

template <typename T>
auto printArray(std::vector<T> const& vec)
{
  std::stringstream ss;
  ss << "[" << vec[0];
  for (auto i = 1u; i < vec.size(); ++i) {
    ss << ", " << vec[i];
  }
  ss << "]";
  return ss.str();
}

template <typename T>
auto printMatrix(Array2D<T> const& m)
{
  std::stringstream ss;
  ss << "[[" << m(0, 0);
  for (auto i = 1u; i < m.cols; ++i) {
    ss << "," << m(0, i);
  }
  for (auto i = 1u; i < m.rows; ++i) {
    ss << "], [" << m(i, 0);
    for (auto j = 1u; j < m.cols; ++j) {
      ss << "," << m(i, j);
    }
  }
  ss << "]]";
  return ss.str();
}

static constexpr float defaultm[3][4] = {{1.1, 1.2, 1.3, 1.4}, {2.1, 2.2, 2.3, 2.4}, {3.1, 3.2, 3.3, 3.4}};
static LabeledArray<float> la{&defaultm[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};

struct ConfigurableObjectDemo {
  Configurable<configurableCut> cut{"cut", {0.5, 1, true}, "generic cut"};
  MutableConfigurable<configurableCut> mutable_cut{"mutable_cut", {1., 2, false}, "generic cut"};

  // note that size is fixed by this declaration - externally supplied vector needs to be the same size!
  Configurable<std::vector<int>> array{"array", {0, 0, 0, 0, 0, 0, 0}, "generic array"};
  Configurable<Array2D<float>> vmatrix{"matrix", {&defaultm[0][0], 3, 4}, "generic matrix"};
  Configurable<LabeledArray<float>> vla{"vla", {defaultm[0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}}, "labeled array"};

  void init(InitContext const&){};
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Cut1: %.3f; Cut2: %.3f", cut, mutable_cut);
    LOGF(INFO, "Cut1 bins: %s; Cut2 bins: %s", printArray(cut->getBins()), printArray(mutable_cut->getBins()));
    LOGF(INFO, "Cut1 labels: %s; Cut2 labels: %s", printArray(cut->getLabels()), printArray(mutable_cut->getLabels()));
    auto vec = (std::vector<int>)array;
    LOGF(INFO, "Array: %s", printArray(vec).c_str());
    LOGF(INFO, "Matrix: %s", printMatrix((Array2D<float>)vmatrix));
    for (auto const& track : tracks) {
      if (track.globalIndex() % 500 == 0) {
        std::string decision1;
        std::string decision2;
        if (cut->method(std::abs(track.eta()))) {
          decision1 = "true";
        } else {
          decision1 = "false";
        }
        if (mutable_cut->method(std::abs(track.eta()))) {
          decision2 = "true";
        } else {
          decision2 = "false";
        }
        LOGF(INFO, "Cut1: %s; Cut2: %s", decision1, decision2);
        if (decision2 == "false") {
          mutable_cut->setState(-1);
        } else {
          mutable_cut->setState(1);
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ConfigurableObjectDemo>("configurable-demo")};
}
