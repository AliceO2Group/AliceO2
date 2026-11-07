// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckTopologyDictionary.C
/// \brief Builds the TRK or FT3 topology dictionary from MC timeframes and checks it on others
///
/// Each timeframe directory holds the clusters with their MC labels
/// (o2clus_trk.root, o2clus_ft3.root), the signal hits (sgn_HitsTRK.root,
/// sgn_HitsFT3.root) and the geometry. A signal cluster is taken with its
/// particle's crossing of the chip, except at the matrix's edge (it may have
/// lost pixels) and where the particle left more clusters on the chip than it
/// crossed it (pieces of one crossing).
///
/// With training timeframes the dictionary is built from them and written to
/// \a dictionaryFile, else read from it. On the check timeframes it prints per
/// layer the pulls residual / width without and with the offsets (median and
/// normalised MAD) and writes per entry the residuals' count, median and
/// width beside the entry's (tree "entries" in \a output).
///
///   root -b -q 'CheckTopologyDictionary.C+("TRK", "TRKTopologyDictionary.root", {"tf7", "tf8"}, {"tf2", "tf3"})'

#define ENABLE_UPGRADES // DetID::TRK and FT3; the macro is not compiled with O2's definitions

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DataFormatsTRKFT3/Cluster.h"
#include "DataFormatsTRKFT3/Hit.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "FT3Base/GeometryTGeo.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/SegmentationChip.h"
#include "TRKBase/Specs.h"
#include "TRKReconstruction/BuildTopologyDictionary.h"
#include <TFile.h>
#include <TTree.h>
#include <fairlogger/Logger.h>
#include <fmt/format.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#endif

namespace check_dictionary
{

using o2::math_utils::Point3D;

template <int DetID>
auto& geometry()
{
  if constexpr (DetID == o2::detectors::DetID::TRK) {
    return *o2::trk::GeometryTGeo::Instance();
  } else {
    return *o2::ft3::GeometryTGeo::Instance();
  }
}

/// The median and normalised MAD of \a values, which it reorders.
std::pair<float, float> medianAndWidth(std::vector<float>& values)
{
  const auto median = [](std::vector<float>& v) {
    const auto middle = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), middle, v.end());
    return v.size() % 2 == 1 ? *middle : 0.5f * (*std::max_element(v.begin(), middle) + *middle);
  };
  if (values.empty()) {
    return {0.f, 0.f};
  }
  const float m = median(values);
  std::vector<float> deviations(values.size());
  std::ranges::transform(values, deviations.begin(), [m](float v) { return std::abs(v - m); });
  return {m, 1.4826f * median(deviations)};
}

/// The o2sim tree of \a path, held open by \a file.
TTree& treeOf(std::unique_ptr<TFile>& file, const std::string& path)
{
  file.reset(TFile::Open(path.c_str()));
  auto* tree = file && !file->IsZombie() ? file->Get<TTree>("o2sim") : nullptr;
  if (tree == nullptr) {
    throw std::runtime_error(fmt::format("{}: no o2sim tree", path));
  }
  return *tree;
}

/// Reads entry 0 of \a tree's branch \a name into \a object.
template <typename T>
void readBranch(TTree& tree, const std::string& name, T& object)
{
  auto* branch = tree.GetBranch(name.c_str());
  if (branch == nullptr) {
    throw std::runtime_error(fmt::format("no branch {}", name));
  }
  auto* pointer = &object;
  branch->SetAddress(&pointer);
  branch->GetEntry(0);
  branch->ResetAddress();
}

/// Calls \a f(cluster, pattern, start, end) for the selected signal clusters
/// of \a directory, with their particle's crossing in O2's local frame.
template <int DetID, typename F>
void forEachCluster(const std::string& directory, F&& f)
{
  using Dictionary = o2::trkft3::TopologyDictionary<DetID>;
  using ClusterType = o2::trkft3::Cluster<DetID>;
  const std::string name = o2::detectors::DetID(DetID).getName();
  struct Segment {
    int track, chip;
    Point3D<float> start, end; ///< global frame
  };
  const auto keyOf = [](const Segment& segment) { return std::pair{segment.track, segment.chip}; };

  // The signal hits, by event, sorted by particle and chip.
  std::vector<std::vector<Segment>> events;
  {
    std::unique_ptr<TFile> file;
    auto& tree = treeOf(file, fmt::format("{}/sgn_Hits{}.root", directory, name));
    std::vector<o2::trkft3::Hit> hits, *hitsPointer = &hits;
    tree.SetBranchAddress(fmt::format("{}Hit", name).c_str(), &hitsPointer);
    events.resize(tree.GetEntries());
    for (std::size_t event = 0; event < events.size(); ++event) {
      tree.GetEntry(static_cast<Long64_t>(event));
      for (const auto& hit : hits) {
        events[event].push_back({hit.GetTrackID(), hit.GetDetectorID(), {hit.GetStartX(), hit.GetStartY(), hit.GetStartZ()}, {hit.GetX(), hit.GetY(), hit.GetZ()}});
      }
      std::ranges::sort(events[event], {}, keyOf);
    }
    tree.ResetBranchAddresses();
  }

  std::unique_ptr<TFile> file;
  auto& tree = treeOf(file, fmt::format("{}/o2clus_{}.root", directory, DetID == o2::detectors::DetID::TRK ? "trk" : "ft3"));
  for (int stream = 0; tree.GetBranch(fmt::format("{}ClusterComp_{}", name, stream).c_str()) != nullptr; ++stream) {
    std::vector<ClusterType> clusters;
    std::vector<unsigned char> patterns;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;
    readBranch(tree, fmt::format("{}ClusterComp_{}", name, stream), clusters);
    readBranch(tree, fmt::format("{}ClusterPatt_{}", name, stream), patterns);
    readBranch(tree, fmt::format("{}ClusterMCTruth_{}", name, stream), labels);

    // The signal clusters with their particle's hits on the chip; the clusters of each particle and chip.
    struct Candidate {
      const ClusterType* cluster;
      const unsigned char* pattern;
      std::span<const Segment> segments;
      std::tuple<int, int, int> key; ///< event, track, chip
    };
    std::vector<Candidate> candidates;
    std::map<std::tuple<int, int, int>, int> nClusters;
    const unsigned char* next = patterns.data();
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      const auto& cluster = clusters[i];
      const auto* pattern = next;
      next += 2 + ((pattern[0] * pattern[1] + 7) / 8);
      const auto ofCluster = labels.getLabels(i);
      const auto signal = std::find_if(ofCluster.begin(), ofCluster.end(), [](const auto& label) { return label.isValid() && !label.isNoise() && label.getSourceID() == 0; });
      if (signal == ofCluster.end()) {
        continue;
      }
      const auto& segments = events.at(signal->getEventID());
      const auto range = std::ranges::equal_range(segments, std::pair{signal->getTrackID(), static_cast<int>(cluster.chipID)}, {}, keyOf);
      if (range.empty()) {
        continue;
      }
      candidates.push_back({&cluster, pattern, {range.begin(), range.end()}, {signal->getEventID(), signal->getTrackID(), cluster.chipID}});
      ++nClusters[candidates.back().key];
    }

    for (const auto& candidate : candidates) {
      const auto& cluster = *candidate.cluster;
      const auto* pattern = candidate.pattern;
      // Left out: at the matrix's edge, or one of several pieces of a crossing.
      const bool vd = cluster.subDetID == 0;
      const int nRows = vd ? o2::trk::SegmentationChip::getNRowsVD(cluster.layer) : o2::trk::constants::moduleMLOT::chip::nRows;
      const int nCols = vd ? o2::trk::constants::VD::petal::layer::nCols : o2::trk::constants::moduleMLOT::chip::nCols;
      if (cluster.row == 0 || cluster.col == 0 || cluster.row + pattern[0] >= nRows || cluster.col + pattern[1] >= nCols ||
          nClusters[candidate.key] > static_cast<int>(candidate.segments.size())) {
        continue;
      }
      // A particle crossing the chip more than once: the crossing whose middle is nearest the centre of gravity.
      const auto& l2g = geometry<DetID>().getMatrixL2G(cluster.chipID);
      const auto cog = Dictionary::centreOfGravity(cluster, pattern);
      float nearest = std::numeric_limits<float>::max();
      Point3D<float> start, end;
      for (const auto& segment : candidate.segments) {
        const auto a = l2g ^ segment.start, b = l2g ^ segment.end;
        const auto fa = Dictionary::localToFlat(cluster, a), fb = Dictionary::localToFlat(cluster, b);
        const float distance = std::hypot(0.5f * (fa.X() + fb.X()) - cog.X(), 0.5f * (fa.Z() + fb.Z()) - cog.Z());
        if (distance < nearest) {
          nearest = distance;
          start = a;
          end = b;
        }
      }
      f(cluster, pattern, start, end);
    }
  }
  LOGP(info, "{}: read", directory);
}

template <int DetID>
void check(const std::string& dictionaryFile, const std::vector<std::string>& timeframes, const std::vector<std::string>& training, const std::string& output,
           bool applyOffsets, int nThreads)
{
  using Dictionary = o2::trkft3::TopologyDictionary<DetID>;
  const auto& directories = training.empty() ? timeframes : training;
  if (directories.empty()) {
    throw std::invalid_argument("no timeframe directories");
  }
  o2::base::GeometryManager::loadGeometry(directories.front() + "/o2sim_geometry-aligned.root");
  geometry<DetID>().fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  std::unique_ptr<Dictionary> dictionary;
  if (!training.empty()) {
    int nLayers = 0;
    if constexpr (DetID == o2::detectors::DetID::TRK) {
      nLayers = o2::trk::constants::VD::petal::nLayers + geometry<DetID>().getNumberOfLayersMLOT();
    } else {
      nLayers = geometry<DetID>().getNumberOfLayers();
    }
    o2::trkft3::BuildTopologyDictionary<DetID> builder(nLayers, {.nThreads = nThreads});
    for (const auto& directory : training) {
      forEachCluster<DetID>(directory, [&](const auto& cluster, const auto* pattern, const auto& start, const auto& end) { builder.accountCluster(cluster, pattern, start, end); });
    }
    builder.build();
    builder.setApplyOffsets(applyOffsets);
    builder.saveDictionaryRoot(dictionaryFile);
    dictionary = std::make_unique<Dictionary>(builder.getDictionary());
    LOGP(info, "written to {}", dictionaryFile);
  } else {
    dictionary.reset(Dictionary::loadFrom(dictionaryFile));
  }
  dictionary->print();
  if (timeframes.empty()) {
    return;
  }
  const bool applied = dictionary->getApplyOffsets();

  // The residuals hit - centre of gravity at the layer's depth, by layer and entry, across and along.
  const int nLayers = dictionary->getNumberOfLayers();
  dictionary->setApplyOffsets(false);
  std::vector<std::map<int, std::array<std::vector<float>, 2>>> residuals(nLayers);
  for (const auto& directory : timeframes) {
    forEachCluster<DetID>(directory, [&](const auto& cluster, const auto* pattern, const auto& start, const auto& end) {
      const int layer = Dictionary::layerOf(cluster);
      if (layer < 0 || layer >= nLayers) {
        throw std::runtime_error(fmt::format("{}: on no layer of the dictionary's {}", cluster.asString(), nLayers));
      }
      const int id = dictionary->findID(layer, pattern);
      const auto position = dictionary->getFlatCoordinates(cluster, pattern, id);
      const auto a = Dictionary::localToFlat(cluster, start), b = Dictionary::localToFlat(cluster, end);
      const float t = b.Y() != a.Y() ? (position.Y() - a.Y()) / (b.Y() - a.Y()) : -1.f;
      if (t >= 0.f && t <= 1.f) {
        auto& entry = residuals[layer][id];
        entry[0].push_back(a.X() + (t * (b.X() - a.X())) - position.X());
        entry[1].push_back(a.Z() + (t * (b.Z() - a.Z())) - position.Z());
      }
    });
  }

  const std::unique_ptr<TFile> out{TFile::Open(output.c_str(), "RECREATE")};
  auto* entries = new TTree("entries", "per layer and entry, across (x) and along (z): the residuals against the entry [cm]");
  int tLayer = 0, tId = 0, tCount = 0;
  bool tGroup = false;
  std::array<float, 2> tMedian{}, tWidth{}, tEntryOffset{}, tEntryWidth{};
  entries->Branch("layer", &tLayer);
  entries->Branch("id", &tId);
  entries->Branch("isGroup", &tGroup);
  entries->Branch("count", &tCount);
  for (int axis = 0; axis < 2; ++axis) {
    const char* a = axis == 0 ? "x" : "z";
    entries->Branch(fmt::format("{}Median", a).c_str(), &tMedian[axis]);
    entries->Branch(fmt::format("{}Width", a).c_str(), &tWidth[axis]);
    entries->Branch(fmt::format("{}EntryOffset", a).c_str(), &tEntryOffset[axis]);
    entries->Branch(fmt::format("{}EntryWidth", a).c_str(), &tEntryWidth[axis]);
  }
  LOGP(info, "pulls residual / width per layer, median and normalised MAD, without | with the offsets (the dictionary's: {})", applied ? "applied" : "not applied");
  for (int layer = 0; layer < nLayers; ++layer) {
    std::array<std::vector<float>, 2> pulls, offsetPulls;
    for (auto& [id, values] : residuals[layer]) {
      const auto& entry = dictionary->getEntry(layer, id);
      const std::array<float, 2> width{entry.xWidth, entry.zWidth}, offset{entry.xOffset, entry.zOffset};
      for (int axis = 0; axis < 2; ++axis) {
        for (const float r : values[axis]) {
          pulls[axis].push_back(r / width[axis]);
          offsetPulls[axis].push_back((r - offset[axis]) / width[axis]);
        }
        std::tie(tMedian[axis], tWidth[axis]) = medianAndWidth(values[axis]);
        tEntryOffset[axis] = offset[axis];
        tEntryWidth[axis] = width[axis];
      }
      tLayer = layer;
      tId = id;
      tGroup = entry.isGroup;
      tCount = static_cast<int>(values[0].size());
      entries->Fill();
    }
    const auto [x, xWidth] = medianAndWidth(pulls[0]);
    const auto [z, zWidth] = medianAndWidth(pulls[1]);
    const auto [xo, xoWidth] = medianAndWidth(offsetPulls[0]);
    const auto [zo, zoWidth] = medianAndWidth(offsetPulls[1]);
    LOGP(info, "  layer {:2}, {:9} residuals: across {:6.3f} {:5.3f} | {:6.3f} {:5.3f}, along {:6.3f} {:5.3f} | {:6.3f} {:5.3f}", layer, pulls[0].size(), x, xWidth, xo,
         xoWidth, z, zWidth, zo, zoWidth);
  }
  out->Write();
  LOGP(info, "written to {}", output);
}

} // namespace check_dictionary

/// \a detector "TRK" or "FT3"; \a timeframes the directories to check on, \a training those to build from (none: \a dictionaryFile is read);
/// \a applyOffsets whether a built dictionary adds the entries' median offsets; \a nThreads the depth scan's.
void CheckTopologyDictionary(const std::string& detector, const std::string& dictionaryFile, const std::vector<std::string>& timeframes,
                             const std::vector<std::string>& training = {}, const std::string& output = "CheckTopologyDictionary.root", bool applyOffsets = false,
                             int nThreads = 8)
{
  if (detector == "TRK") {
    check_dictionary::check<o2::detectors::DetID::TRK>(dictionaryFile, timeframes, training, output, applyOffsets, nThreads);
  } else if (detector == "FT3") {
    check_dictionary::check<o2::detectors::DetID::FT3>(dictionaryFile, timeframes, training, output, applyOffsets, nThreads);
  } else {
    LOGP(fatal, "unknown detector {}: TRK or FT3", detector);
  }
}
