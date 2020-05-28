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
#include <gsl/gsl>
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

    // Fill the ClusterNative access structure from data and corresponding mc label arrays
    // from the internal data structures of the reader.
    int fillIndex(ClusterNativeAccess& clusterIndex, std::unique_ptr<ClusterNative[]>& clusterBuffer,
                  MCLabelContainer& mcBuffer);

    // Fill the ClusterNative access structure from data and corresponding mc label arrays.
    // Both cluster data input and mc containers are provided as a collection with one entry per
    // sector. The data per sector itself is again a collection.
    //
    // MC truth per sector is organized as one MCLabelContainer per row.
    //
    // The index structure does not own the data, specific buffers owned by the caller must be
    // provided for both clusters and mc labels.
    //
    // FIXME: while this function was originally indendet to fill the index only and leaver any
    // data as is, commit 65e17cb73e (PR2166) introduces a rearrangement of data which probably
    // should be moved to a separate function for clarity. Maybe another access index named
    // ClusterNativeMonAccess is more appropriate. Also it probably makes sense to performe the
    // linarization already in the decoder and abandon the partitioning of data pad-row wise.
    //
    // @param clusterIndex   the target where all the pointers are set
    // @param clusterBuffer  array of ClusterNative, clusters are copied to consecutive sections
    //                       pointers in the index point to regions in this buffer
    // @param mcBuffer
    // @param inputs         data arrays, fixed array, one per sector
    // @param mcinputs       vectors mc truth container, fixed array, one per sector
    // @param checkFct       check whether a sector index is valid
    template <typename DataArrayType, typename MCArrayType, typename CheckFct>
    static int fillIndex(
      ClusterNativeAccess& clusterIndex, std::unique_ptr<ClusterNative[]>& clusterBuffer,
      MCLabelContainer& mcBuffer, DataArrayType& inputs, MCArrayType const& mcinputs,
      CheckFct checkFct = [](auto const&) { return true; });

    template <typename DataArrayType, typename CheckFct = std::function<bool(size_t&)>>
    static int fillIndex(
      ClusterNativeAccess& clusterIndex, std::unique_ptr<ClusterNative[]>& clusterBuffer,
      DataArrayType& inputs, CheckFct checkFct = [](auto const&) { return true; })
    {
      // just use a dummy parameter with empty vectors, the ugly double vector will be removed
      // soon. TODO: maybe do in one function with conditional template parameter
      std::vector<std::vector<MCLabelContainer>> dummy(inputs.size());
      // another default, nothing will be added to the container
      MCLabelContainer mcBuffer;
      return fillIndex(clusterIndex, clusterBuffer, mcBuffer, inputs, dummy, checkFct);
    }

    // Process data for one sector.
    // This function does not copy any data but sets the corresponding poiters in the index.
    // Cluster data are provided as a raw buffer of consecutive ClusterNative arrays preceded by ClusterGroupHeader
    // MC labels are provided as a span of MCLabelContainers, one per pad row.
    static int parseSector(const char* buffer, size_t size, gsl::span<MCLabelContainer const> const& mcinput, //
                           ClusterNativeAccess& clusterIndex,                                                 //
                           const MCLabelContainer* (&clustersMCTruth)[NSectors][NPadRows]);                   //

    // Process data for one sector
    // Helper method receiving raw buffer provided as container
    // This function does not copy any data but sets the corresponding poiters in the index.
    template <typename ContainerT>
    static int parseSector(ContainerT const cont, gsl::span<MCLabelContainer const> const& mcinput, //
                           ClusterNativeAccess& clusterIndex,                                       //
                           const MCLabelContainer* (&clustersMCTruth)[NSectors][NPadRows])          //
    {
      using T = typename std::remove_pointer<ContainerT>::type;
      static_assert(sizeof(typename T::value_type) == 1, "raw container must be byte-type");
      T const* container = nullptr;
      if constexpr (std::is_pointer<ContainerT>::value) {
        if (cont == nullptr) {
          return 0;
        }
        container = cont;
      } else {
        container = &cont;
      }
      return parseSector(container->data(), container->size(), mcinput, clusterIndex, clustersMCTruth);
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

  /// copy data of the specified sector from the index to a byte-type container
  /// optional MC labels are separated accordingly and added to a target vector.
  /// @param index     the cluster index object
  /// @param target    a container object, will be resized accordingly
  /// @param mcTarget  container to receive the separated MC label objects
  template <typename BufferType, typename MCArrayType>
  static void copySectorData(ClusterNativeAccess const& index, int sector, BufferType& target, MCArrayType& mcTarget);
};

template <typename DataArrayType, typename MCArrayType, typename CheckFct>
int ClusterNativeHelper::Reader::fillIndex(ClusterNativeAccess& clusterIndex,
                                           std::unique_ptr<ClusterNative[]>& clusterBuffer, MCLabelContainer& mcBuffer,
                                           DataArrayType& inputs, MCArrayType const& mcinputs, CheckFct checkFct)
{
  if (mcinputs.size() > 0 && mcinputs.size() != inputs.size()) {
    std::runtime_error("inconsistent size of MC label array " + std::to_string(mcinputs.size()) + ", expected " + std::to_string(inputs.size()));
  }
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  const MCLabelContainer* clustersMCTruth[NSectors][NPadRows] = {};
  int result = 0;
  for (size_t index = 0, end = inputs.size(); index < end; index++) {
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

template <typename BufferType, typename MCArrayType>
void ClusterNativeHelper::copySectorData(ClusterNativeAccess const& index, int sector, BufferType& target, MCArrayType& mcTarget)
{
  static_assert(sizeof(typename BufferType::value_type) == 1, "Target container must be byte-type");
  if (index.clustersLinear == nullptr) {
    return;
  }
  size_t nRows = 0;
  size_t nClusters = 0;
  for (unsigned int row = 0; row < NPadRows; row++) {
    // count rows with clusters
    nRows += index.nClusters[sector][row] > 0 ? 1 : 0;
    nClusters += index.nClusters[sector][row];
  }
  size_t rawSize = nRows * sizeof(ClusterNativeBuffer) + nClusters * sizeof(ClusterNative);
  target.resize(rawSize);
  ClusterNativeBuffer* current = reinterpret_cast<ClusterNativeBuffer*>(target.data());
  for (unsigned int row = 0; row < NPadRows; row++) {
    if (index.nClusters[sector][row] == 0) {
      continue;
    }
    current->sector = sector;
    current->globalPadRow = row;
    current->nClusters = index.nClusters[sector][row];
    memcpy(current->clusters, index.clusters[sector][row], sizeof(*(current->clusters)) * current->nClusters);
    current = reinterpret_cast<ClusterNativeBuffer*>(current->clusters + current->nClusters);
    if (index.clustersMCTruth) {
      mcTarget.emplace_back();
      for (unsigned int k = 0; k < index.nClusters[sector][row]; k++) {
        for (auto const& label : index.clustersMCTruth->getLabels(k + index.clusterOffset[sector][row])) {
          mcTarget.back().addElement(k, label);
        }
      }
    }
  }
}

} // namespace tpc
} // namespace o2
#endif // CLUSTERNATIVEHELPER_H
