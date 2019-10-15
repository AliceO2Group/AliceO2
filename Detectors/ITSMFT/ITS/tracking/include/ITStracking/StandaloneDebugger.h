// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \fileStandaloneDebugger.h
/// \brief separate TreeStreamerRedirector class to be used with GPU
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_STANDALONE_DEBUGGER_H_
#define O2_ITS_STANDALONE_DEBUGGER_H_

namespace o2
{

class MCCompLabel;

namespace utils
{
class TreeStreamRedirector;
}

namespace its
{
class Tracklet;
class Line;
class ROframe;

class StandaloneDebugger
{
 public:
  explicit StandaloneDebugger(const std::string debugTreeFileName = "dbg_ITS.root");
  ~StandaloneDebugger();
  void setDebugTreeFileName(std::string);
  const std::string& getDebugTreeFileName() const { return mDebugTreeFileName; }

  void fillCombinatoricsTree(std::vector<Tracklet>, std::vector<Tracklet>);
  void fillCombinatoricsMCTree(std::vector<Tracklet>, std::vector<Tracklet>);
  void fillTrackletSelectionTree(std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>&,
                                 std::vector<Tracklet> comb01,
                                 std::vector<Tracklet> comb12,
                                 std::vector<std::array<int, 2>>,
                                 const ROframe*);
  void fillLinesSummaryTree(std::vector<Line>, const ROframe*);
  void fillPairsInfoTree(std::vector<Line>, const ROframe*);
  void fillXYZHistogramTree(std::array<std::vector<int>, 3>, const std::array<int, 3>);

 private:
  std::string mDebugTreeFileName = "dbg_ITS.root"; // output filename
  o2::utils::TreeStreamRedirector* mTreeStream;    // observer
};

inline void StandaloneDebugger::setDebugTreeFileName(const std::string name)
{
  if (!name.empty()) {
    mDebugTreeFileName = name;
  }
}

} // namespace its
} // namespace o2

#endif /*O2_ITS_STANDALONE_DEBUGGER_H_*/