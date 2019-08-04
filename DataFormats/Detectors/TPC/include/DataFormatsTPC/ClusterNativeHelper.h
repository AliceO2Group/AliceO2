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
namespace tpc
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

  size_t getFlatSize() const { return sizeof(attribute_type) + clusters.size() * sizeof(value_type); }

  const value_type* data() const { return clusters.data(); }

  value_type* data() { return clusters.data(); }

  std::vector<ClusterNative> clusters;
};

/// @struct ClusterNativeBuffer
/// Contiguous buffer for a collection of ClusterNative objects
/// belonging to a row.
/// The struct inherits the sector, globalPadRow, and nClusters members from the property
/// ClusterGroupHeader.
///
/// Used for messages
///
struct ClusterNativeBuffer : public ClusterGroupHeader {
  using attribute_type = ClusterGroupHeader;
  using value_type = ClusterNative;

  size_t getFlatSize() const { return sizeof(attribute_type) + nClusters * sizeof(value_type); }

  const value_type* data() const { return clusters; }

  value_type* data() { return clusters; }

  value_type clusters[0];
};

/// @class ClusterNativeHelper utility class for TPC native clusters
/// This class supports the following utility functionality for handling of
/// TPC ClusterNative data:
/// - interface to the ClusterNativeAccess cluster access index
/// - reading of ClusterNative data in binary format
/// - conversion to a tree structure for easy examination of the cluster parameters
///
/// The class adds a Reader for the binary format of decoded native clusters as
/// written by the TPC reconstruction workflow. The reader fills the access index
/// ClusterNativeAccess in the first version. We can think of something
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

  constexpr static unsigned int NSectors = o2::tpc::Constants::MAXSECTOR;
  constexpr static unsigned int NPadRows = o2::tpc::Constants::MAXGLOBALPADROW;

  /// convert clusters stored in binary cluster native format to a tree and write to root file
  /// the cluster parameters are stored in the tree together with sector and padrow numbers.
  static void convert(const char* fromFile, const char* toFile, const char* toTreeName = "tpcnative");

  // Helper function to create a ClusterNativeAccess structure from a std::vector of ClusterNative containers
  // This is not contained in the ClusterNative class itself to reduce the dependencies of the class
  static std::unique_ptr<ClusterNativeAccess> createClusterNativeIndex(
    std::unique_ptr<ClusterNative[]>& buffer, std::vector<ClusterNativeContainer>& clusters,
    MCLabelContainer* bufferMC = nullptr,
    std::vector<MCLabelContainer>* mcTruth = nullptr);

  // add clusters from a flattened buffer starting with an attribute, e.g. ClusterGroupAttribute followed
  // by the array of ClusterNative, the number of clusters is determined from the size of the buffer
  // FIXME: add mc labels
  template <typename AttributeT>
  static int addFlatBuffer(ClusterNativeAccess& clusterIndex, unsigned char* buffer, size_t size)
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

    int fillIndex(ClusterNativeAccess& clusterIndex, std::unique_ptr<ClusterNative[]>& clusterBuffer,
                  MCLabelContainer& mcBuffer);

    template <typename DataArrayType, typename MCArrayType, typename CheckFct>
    static int fillIndex(
      ClusterNativeAccess& clusterIndex, std::unique_ptr<ClusterNative[]>& clusterBuffer,
      MCLabelContainer& mcBuffer, DataArrayType& inputs, MCArrayType& mcinputs,
      CheckFct checkFct = [](auto&) { return true; });

    static int parseSector(const char* buffer, size_t size, std::vector<MCLabelContainer>& mcinput, ClusterNativeAccess& clusterIndex,
                           const MCLabelContainer* (&clustersMCTruth)[NSectors][NPadRows]);
    template <typename ContainerT, std::enable_if_t<std::is_pointer<ContainerT>::value, int> = 0>
    static int parseSector(ContainerT container, std::vector<MCLabelContainer>& mcinput, ClusterNativeAccess& clusterIndex,
                           const MCLabelContainer* (&clustersMCTruth)[NSectors][NPadRows])
    {
      if (container == nullptr) {
        return 0;
      }
      return parseSector(container->data(), container->size(), mcinput, clusterIndex, clustersMCTruth);
    }
    template <typename ContainerT, std::enable_if_t<!std::is_pointer<ContainerT>::value, int> = 0>
    static int parseSector(ContainerT container, std::vector<MCLabelContainer>& mcinput, ClusterNativeAccess& clusterIndex,
                           const MCLabelContainer* (&clustersMCTruth)[NSectors][NPadRows])
    {
      return parseSector(container.data(), container.size(), mcinput, clusterIndex, clustersMCTruth);
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
    std::array<std::vector<char>*, NSectors> mSectorRaw = {nullptr};
    /// the array of raw buffers
    std::array<size_t, NSectors> mSectorRawSize = {0};
    /// the array of MC label containers
    std::array<std::vector<MCLabelContainer>, NSectors> mSectorMC;
    /// pointers on the elements of array of MC label containers
    std::array<std::vector<MCLabelContainer>*, NSectors> mSectorMCPtr = {nullptr};
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
    int fillFrom(ClusterNativeAccess const& clusterIndex);
    /// fill tree from a single cluster array
    int fillFrom(int sector, int padrow, ClusterNative const* clusters, size_t nClusters,
                 MCLabelContainer* = nullptr);

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

template <typename DataArrayType, typename MCArrayType, typename CheckFct>
int ClusterNativeHelper::Reader::fillIndex(ClusterNativeAccess& clusterIndex,
                                           std::unique_ptr<ClusterNative[]>& clusterBuffer, MCLabelContainer& mcBuffer,
                                           DataArrayType& inputs, MCArrayType& mcinputs, CheckFct checkFct)
{
  static_assert(std::tuple_size<DataArrayType>::value == std::tuple_size<MCArrayType>::value);
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  const MCLabelContainer* clustersMCTruth[NSectors][NPadRows] = {};
  int result = 0;
  for (size_t index = 0; index < NSectors; index++) {
    if (!checkFct(index)) {
      continue;
    }
    int locres = parseSector(inputs[index], mcinputs[index], clusterIndex, clustersMCTruth);
    if (locres < 0) {
      return locres;
    }
    result += locres;
  }

  // Now move all data to a new consecutive buffer
  ClusterNativeAccess old = clusterIndex;
  clusterBuffer.reset(new ClusterNative[result]);
  mcBuffer.clear();
  bool mcPresent = false;
  clusterIndex.clustersLinear = clusterBuffer.get();
  clusterIndex.setOffsetPtrs();
  for (unsigned int i = 0; i < NSectors; i++) {
    for (unsigned int j = 0; j < NPadRows; j++) {
      memcpy(&clusterBuffer[clusterIndex.clusterOffset[i][j]], old.clusters[i][j], sizeof(*old.clusters[i][j]) * old.nClusters[i][j]);
      if (clustersMCTruth[i][j]) {
        mcPresent = true;
        for (unsigned int k = 0; k < old.nClusters[i][j]; k++) {
          for (auto const& label : clustersMCTruth[i][j]->getLabels(k)) {
            mcBuffer.addElement(clusterIndex.clusterOffset[i][j] + k, label);
          }
        }
      }
    }
  }
  if (mcPresent) {
    clusterIndex.clustersMCTruth = &mcBuffer;
  }

  return result;
}

} // namespace tpc
} // namespace o2
#endif // CLUSTERNATIVEHELPER_H
