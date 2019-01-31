// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef CLUSTERNATIVEHELPER_H
#define CLUSTERNATIVEHELPER_H
/// @file ClusterNativeHelper.h
/// @brief Helper class to read the binary format of TPC ClusterNative
/// @since 2019-01-23
/// @author Matthias Richter

#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TFile.h>
#include <TTree.h>
#include <array>
#include <vector>
#include <string>
#include <tuple> //std::tuple_size
#include <type_traits>

namespace o2
{
namespace TPC
{
using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

/// @struct ClusterNativeContainer
/// A container class for a collection of ClusterNative object
/// belonging to a row.
/// The struct inherits the sector and globalPadRow members of ClusterGroupAttribute.
///
/// Not for permanent storage.
///
struct ClusterNativeContainer : public ClusterGroupAttribute {
  using attribute_type = ClusterGroupAttribute;
  using value_type = ClusterNative;

  static bool sortComparison(const ClusterNative& a, const ClusterNative& b)
  {
    if (a.getTimePacked() != b.getTimePacked()) {
      return (a.getTimePacked() < b.getTimePacked());
    } else {
      return (a.padPacked < b.padPacked);
    }
  }

  size_t getFlatSize() const { return sizeof(attribute_type) + clusters.size() * sizeof(value_type); }

  const value_type* data() const { return clusters.data(); }

  value_type* data() { return clusters.data(); }

  std::vector<ClusterNative> clusters;
};

/// @class ClusterNativeHelper utility class for TPC native clusters
/// This class supports the following utility functionality for handling of
/// TPC ClusterNative data:
/// - interface to the ClusterNativeAccessFullTPC cluster access index
/// - reading of ClusterNative data in binary format
/// - conversion to a tree structure for easy examination of the cluster parameters
///
/// The class adds a Reader for the binary format of decoded native clusters as
/// written by the TPC reconstruction workflow. The reader fills the access index
/// ClusterNativeAccessFullTPC in the first version. We can think of something
/// smarter later.
///
/// The Writer class converts data from a cluster index to a ROOT tree which then
/// allows to inspect the parameters of clusters.
///
/// Finally, ClusterNativeHelper::convert("from.root", "to.root") combines the two.
class ClusterNativeHelper
{
 public:
  ClusterNativeHelper() = default;
  ~ClusterNativeHelper() = default;

  constexpr static size_t NSectors = o2::TPC::Constants::MAXSECTOR;

  /// convert clusters stored in binary cluster native format to a tree and write to root file
  /// the cluster parameters are stored in the tree together with sector and padrow numbers.
  static void convert(const char* fromFile, const char* toFile, const char* toTreeName = "tpcnative");

  // Helper function to create a ClusterNativeAccessFullTPC structure from a std::vector of ClusterNative containers
  // This is not contained in the ClusterNative class itself to reduce the dependencies of the class
  static std::unique_ptr<ClusterNativeAccessFullTPC> createClusterNativeIndex(
    std::vector<ClusterNativeContainer>& clusters,
    std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* mcTruth = nullptr);

  // add clusters from a flattened buffer starting with an attribute, e.g. ClusterGroupAttribute followed
  // by the array of ClusterNative, the number of clusters is determined from the size of the buffer
  // FIXME: add mc labels
  template <typename AttributeT>
  static int addFlatBuffer(ClusterNativeAccessFullTPC& clusterIndex, unsigned char* buffer, size_t size)
  {
    if (buffer == nullptr || size < sizeof(AttributeT) || (size - sizeof(AttributeT)) % sizeof(ClusterNative) != 0) {
      // this is not a valid message, incompatible size
      return -1;
    }
    const auto& groupAttribute = *reinterpret_cast<AttributeT*>(buffer);
    auto nofClusters = (size - sizeof(AttributeT)) / sizeof(ClusterNative);
    auto ptrClusters = reinterpret_cast<ClusterNative*>(buffer + sizeof(groupAttribute));
    clusterIndex.clusters[groupAttribute.sector][groupAttribute.globalPadRow] = ptrClusters;
    clusterIndex.nClusters[groupAttribute.sector][groupAttribute.globalPadRow] = nofClusters;
    return nofClusters;
  }

  /// @class Reader
  /// @brief A reader class for the raw cluster native data
  ///
  class Reader
  {
   public:
    Reader();
    ~Reader();

    void init(const char* filename, const char* treename = nullptr);
    size_t getTreeSize() const
    {
      return (mTree ? mTree->GetEntries() : 0);
    }
    void read(size_t entry);
    void clear();

    int fillIndex(ClusterNativeAccessFullTPC& clusterIndex);

    template <typename DataArrayType, typename MCArrayType, typename CheckFct>
    static int fillIndex(ClusterNativeAccessFullTPC& clusterIndex,
                         DataArrayType& inputs, MCArrayType& mcinputs,
                         CheckFct checkFct = [](auto&) { return true; })
    {
      static_assert(std::tuple_size<DataArrayType>::value == std::tuple_size<MCArrayType>::value);
      memset(&clusterIndex, 0, sizeof(clusterIndex));
      int result = 0;
      for (size_t index = 0; index < NSectors; index++) {
        if (!checkFct(index)) {
          continue;
        }
        int locres = parseSector(inputs[index], mcinputs[index], clusterIndex);
        if (result >= 0 && locres >= 0) {
          result += locres;
        } else if (result >= 0) {
          result = locres;
        }
      }
      return result;
    }

    static int parseSector(const char* buffer, size_t size, std::vector<MCLabelContainer>& mcinput, ClusterNativeAccessFullTPC& clusterIndex);
    template <typename ContainerT, std::enable_if_t<std::is_pointer<ContainerT>::value, int> = 0>
    static int parseSector(ContainerT container, std::vector<MCLabelContainer>& mcinput, ClusterNativeAccessFullTPC& clusterIndex)
    {
      if (container == nullptr) {
        return 0;
      }
      return parseSector(container->data(), container->size(), mcinput, clusterIndex);
    }
    template <typename ContainerT, std::enable_if_t<!std::is_pointer<ContainerT>::value, int> = 0>
    static int parseSector(ContainerT container, std::vector<MCLabelContainer>& mcinput, ClusterNativeAccessFullTPC& clusterIndex)
    {
      return parseSector(container.data(), container.size(), mcinput, clusterIndex);
    }

   private:
    /// name of the tree
    std::string mTreeName = "tpcrec";
    /// the base name for the data branches
    std::string mDataBranchName = "TPCClusterNative";
    /// the base name for label branches
    std::string mMCBranchName = "TPCClusterNativeMCTruth";

    /// file instance
    std::unique_ptr<TFile> mFile;
    /// tree
    TTree* mTree = nullptr;
    /// the array of raw buffers
    std::array<std::vector<char>*, NSectors> mSectorRaw;
    /// the array of raw buffers
    std::array<size_t, NSectors> mSectorRawSize;
    /// the array of MC label containers
    std::array<std::vector<MCLabelContainer>, NSectors> mSectorMC;
  };

  /// @class TreeWriter
  /// @brief Utility to write native cluster format to a ROOT tree
  class TreeWriter
  {
   public:
    TreeWriter() = default;
    ~TreeWriter();

    void init(const char* filename, const char* treename);
    void close();

    /// fill tree from the full index of cluster arrays
    int fillFrom(ClusterNativeAccessFullTPC const& clusterIndex);
    /// fill tree from a single cluster array
    int fillFrom(int sector, int padrow, ClusterNative const* clusters, size_t nClusters,
                 o2::dataformats::MCTruthContainer<o2::MCCompLabel>* = nullptr);

    struct BranchData {
      BranchData& operator=(ClusterNative const& rhs)
      {
        time = rhs.getTime();
        pad = rhs.getPad();
        sigmaTime = rhs.getSigmaTime();
        sigmaPad = rhs.getSigmaPad();
        qMax = rhs.qMax;
        qTot = rhs.qTot;
        flags = rhs.getFlags();
        return *this;
      }
      int sector = -1;
      int padrow = -1;
      float time = 0.;
      float pad = 0.;
      float sigmaTime = 0.;
      float sigmaPad = 0.;
      uint16_t qMax = 0;
      uint16_t qTot = 0;
      uint8_t flags = 0;
    };

   private:
    /// file instance
    std::unique_ptr<TFile> mFile;
    /// tree
    std::unique_ptr<TTree> mTree;
    /// cluster store
    std::vector<BranchData> mStoreClusters = {};
    /// the pointer to the store
    std::vector<BranchData>* mStore = &mStoreClusters;
    /// the event counter
    int mEvent = -1;
  };
};
} // namespace TPC
} // namespace o2
#endif // CLUSTERNATIVEHELPER_H
