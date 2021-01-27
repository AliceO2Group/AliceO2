// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTrackingIO.cxx
/// \author David Rohr

#include "GPUChainTracking.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCSliceOutCluster.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTPCTrack.h"
#include "GPUTPCHitId.h"
#include "GPUTRDTrackletWord.h"
#include "AliHLTTPCClusterMCData.h"
#include "GPUTPCMCInfo.h"
#include "GPUTRDTrack.h"
#include "GPUTRDTracker.h"
#include "AliHLTTPCRawCluster.h"
#include "GPUTRDTrackletLabels.h"
#include "GPUDisplay.h"
#include "GPUQA.h"
#include "GPULogging.h"
#include "GPUReconstructionConvert.h"
#include "GPUMemorySizeScalers.h"
#include "GPUTrackingInputProvider.h"

#ifdef HAVE_O2HEADERS
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "GPUTPCClusterStatistics.h"
#include "DataFormatsTPC/ZeroSuppression.h"
#include "GPUHostDataTypes.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCdEdxCalibrationSplines.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#include "TPCFastTransform.h"

#include "utils/linux_helpers.h"

using namespace GPUCA_NAMESPACE::gpu;

#include "GPUO2DataTypes.h"

using namespace o2::tpc;
using namespace o2::trd;
using namespace o2::dataformats;

static constexpr unsigned int DUMP_HEADER_SIZE = 4;
static constexpr char DUMP_HEADER[DUMP_HEADER_SIZE + 1] = "CAv1";

GPUChainTracking::InOutMemory::InOutMemory() = default;
GPUChainTracking::InOutMemory::~InOutMemory() = default;
GPUChainTracking::InOutMemory::InOutMemory(GPUChainTracking::InOutMemory&&) = default;
GPUChainTracking::InOutMemory& GPUChainTracking::InOutMemory::operator=(GPUChainTracking::InOutMemory&&) = default; // NOLINT: False positive in clang-tidy

void GPUChainTracking::DumpData(const char* filename)
{
  FILE* fp = fopen(filename, "w+b");
  if (fp == nullptr) {
    return;
  }
  fwrite(DUMP_HEADER, 1, DUMP_HEADER_SIZE, fp);
  fwrite(&GPUReconstruction::geometryType, sizeof(GPUReconstruction::geometryType), 1, fp);
  DumpData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, InOutPointerType::CLUSTER_DATA);
  DumpData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, InOutPointerType::RAW_CLUSTERS);
#ifdef HAVE_O2HEADERS
  if (mIOPtrs.clustersNative) {
    DumpData(fp, &mIOPtrs.clustersNative->clustersLinear, &mIOPtrs.clustersNative->nClustersTotal, InOutPointerType::CLUSTERS_NATIVE);
    fwrite(&mIOPtrs.clustersNative->nClusters[0][0], sizeof(mIOPtrs.clustersNative->nClusters[0][0]), NSLICES * GPUCA_ROW_COUNT, fp);
    if (mIOPtrs.clustersNative->clustersMCTruth) {
      const auto& buffer = mIOPtrs.clustersNative->clustersMCTruth->getBuffer();
      std::pair<const char*, size_t> tmp = {buffer.data(), buffer.size()};
      DumpData(fp, &tmp.first, &tmp.second, InOutPointerType::CLUSTER_NATIVE_MC);
    }
  }
  if (mIOPtrs.tpcPackedDigits) {
    DumpData(fp, mIOPtrs.tpcPackedDigits->tpcDigits, mIOPtrs.tpcPackedDigits->nTPCDigits, InOutPointerType::TPC_DIGIT);
    if (mIOPtrs.tpcPackedDigits->tpcDigitsMC) {
      const char* ptrs[NSLICES];
      size_t sizes[NSLICES];
      for (unsigned int i = 0; i < NSLICES; i++) {
        const auto& buffer = mIOPtrs.tpcPackedDigits->tpcDigitsMC->v[i]->getBuffer();
        ptrs[i] = buffer.data();
        sizes[i] = buffer.size();
      }
      DumpData(fp, ptrs, sizes, InOutPointerType::TPC_DIGIT_MC);
    }
  }
  if (mIOPtrs.tpcZS) {
    size_t total = 0;
    for (int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[i].count[j]; k++) {
          total += mIOPtrs.tpcZS->slice[i].nZSPtr[j][k];
        }
      }
    }
    std::vector<std::array<char, TPCZSHDR::TPC_ZS_PAGE_SIZE>> pages(total);
    char* ptr = pages[0].data();
    GPUTrackingInOutZS::GPUTrackingInOutZSCounts counts;
    total = 0;
    for (int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        for (unsigned int k = 0; k < mIOPtrs.tpcZS->slice[i].count[j]; k++) {
          memcpy(&ptr[total * TPCZSHDR::TPC_ZS_PAGE_SIZE], mIOPtrs.tpcZS->slice[i].zsPtr[j][k], mIOPtrs.tpcZS->slice[i].nZSPtr[j][k] * TPCZSHDR::TPC_ZS_PAGE_SIZE);
          counts.count[i][j] += mIOPtrs.tpcZS->slice[i].nZSPtr[j][k];
          total += mIOPtrs.tpcZS->slice[i].nZSPtr[j][k];
        }
      }
    }
    total *= TPCZSHDR::TPC_ZS_PAGE_SIZE;
    DumpData(fp, &ptr, &total, InOutPointerType::TPC_ZS);
    fwrite(&counts, sizeof(counts), 1, fp);
  }
#endif
  DumpData(fp, mIOPtrs.sliceTracks, mIOPtrs.nSliceTracks, InOutPointerType::SLICE_OUT_TRACK);
  DumpData(fp, mIOPtrs.sliceClusters, mIOPtrs.nSliceClusters, InOutPointerType::SLICE_OUT_CLUSTER);
  DumpData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, InOutPointerType::MC_LABEL_TPC);
  DumpData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, InOutPointerType::MC_INFO_TPC);
  DumpData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, InOutPointerType::MERGED_TRACK);
  DumpData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
  DumpData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, InOutPointerType::TRD_TRACK);
  DumpData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, InOutPointerType::TRD_TRACKLET);
  DumpData(fp, &mIOPtrs.trdTrackletsMC, &mIOPtrs.nTRDTrackletsMC, InOutPointerType::TRD_TRACKLET_MC);
  fclose(fp);
}

int GPUChainTracking::ReadData(const char* filename)
{
  ClearIOPointers();
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    return (1);
  }

  char buf[DUMP_HEADER_SIZE + 1] = "";
  size_t r = fread(buf, 1, DUMP_HEADER_SIZE, fp);
  if (strncmp(DUMP_HEADER, buf, DUMP_HEADER_SIZE)) {
    GPUError("Invalid file header");
    fclose(fp);
    return -1;
  }
  GeometryType geo;
  r = fread(&geo, sizeof(geo), 1, fp);
  if (geo != GPUReconstruction::geometryType) {
    GPUError("File has invalid geometry (%s v.s. %s)", GPUReconstruction::GEOMETRY_TYPE_NAMES[(int)geo], GPUReconstruction::GEOMETRY_TYPE_NAMES[(int)GPUReconstruction::geometryType]);
    fclose(fp);
    return 1;
  }
  GPUTPCClusterData* ptrClusterData[NSLICES];
  ReadData(fp, mIOPtrs.clusterData, mIOPtrs.nClusterData, mIOMem.clusterData, InOutPointerType::CLUSTER_DATA, ptrClusterData);
  AliHLTTPCRawCluster* ptrRawClusters[NSLICES];
  ReadData(fp, mIOPtrs.rawClusters, mIOPtrs.nRawClusters, mIOMem.rawClusters, InOutPointerType::RAW_CLUSTERS, ptrRawClusters);
  int nClustersTotal = 0;
#ifdef HAVE_O2HEADERS
  mIOMem.clusterNativeAccess.reset(new ClusterNativeAccess);
  if (ReadData<ClusterNative>(fp, &mIOMem.clusterNativeAccess->clustersLinear, &mIOMem.clusterNativeAccess->nClustersTotal, &mIOMem.clustersNative, InOutPointerType::CLUSTERS_NATIVE)) {
    r = fread(&mIOMem.clusterNativeAccess->nClusters[0][0], sizeof(mIOMem.clusterNativeAccess->nClusters[0][0]), NSLICES * GPUCA_ROW_COUNT, fp);
    mIOMem.clusterNativeAccess->setOffsetPtrs();
    mIOPtrs.clustersNative = mIOMem.clusterNativeAccess.get();
    std::pair<const char*, size_t> tmp = {nullptr, 0};
    if (ReadData(fp, &tmp.first, &tmp.second, &mIOMem.clusterNativeMC, InOutPointerType::CLUSTER_NATIVE_MC)) {
      mIOMem.clusterNativeMCView = std::make_unique<ConstMCLabelContainerView>(gsl::span<const char>(tmp.first, tmp.first + tmp.second));
      mIOMem.clusterNativeAccess->clustersMCTruth = mIOMem.clusterNativeMCView.get();
    }
  }
  mIOMem.digitMap.reset(new GPUTrackingInOutDigits);
  if (ReadData(fp, mIOMem.digitMap->tpcDigits, mIOMem.digitMap->nTPCDigits, mIOMem.tpcDigits, InOutPointerType::TPC_DIGIT)) {
    mIOPtrs.tpcPackedDigits = mIOMem.digitMap.get();
    const char* ptrs[NSLICES];
    size_t sizes[NSLICES];
    if (ReadData(fp, ptrs, sizes, mIOMem.tpcDigitsMC, InOutPointerType::TPC_DIGIT_MC)) {
      mIOMem.tpcDigitMCMap = std::make_unique<GPUTPCDigitsMCInput>();
      mIOMem.tpcDigitMCView.reset(new ConstMCLabelContainerView[NSLICES]);
      for (unsigned int i = 0; i < NSLICES; i++) {
        mIOMem.tpcDigitMCView.get()[i] = gsl::span<const char>(ptrs[i], ptrs[i] + sizes[i]);
        mIOMem.tpcDigitMCMap->v[i] = mIOMem.tpcDigitMCView.get() + i;
      }
      mIOMem.digitMap->tpcDigitsMC = mIOMem.tpcDigitMCMap.get();
    }
  }
  const char* ptr;
  size_t total;
  char* ptrZSPages;
  if (ReadData(fp, &ptr, &total, &mIOMem.tpcZSpagesChar, InOutPointerType::TPC_ZS, &ptrZSPages)) {
    GPUTrackingInOutZS::GPUTrackingInOutZSCounts counts;
    r = fread(&counts, sizeof(counts), 1, fp);
    mIOMem.tpcZSmeta.reset(new GPUTrackingInOutZS);
    mIOMem.tpcZSmeta2.reset(new GPUTrackingInOutZS::GPUTrackingInOutZSMeta);
    total = 0;
    for (int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUTrackingInOutZS::NENDPOINTS; j++) {
        mIOMem.tpcZSmeta2->ptr[i][j] = &ptrZSPages[total * TPCZSHDR::TPC_ZS_PAGE_SIZE];
        mIOMem.tpcZSmeta->slice[i].zsPtr[j] = &mIOMem.tpcZSmeta2->ptr[i][j];
        mIOMem.tpcZSmeta2->n[i][j] = counts.count[i][j];
        mIOMem.tpcZSmeta->slice[i].nZSPtr[j] = &mIOMem.tpcZSmeta2->n[i][j];
        mIOMem.tpcZSmeta->slice[i].count[j] = 1;
        total += counts.count[i][j];
      }
    }
    mIOPtrs.tpcZS = mIOMem.tpcZSmeta.get();
  }
#endif
  ReadData(fp, mIOPtrs.sliceTracks, mIOPtrs.nSliceTracks, mIOMem.sliceTracks, InOutPointerType::SLICE_OUT_TRACK);
  ReadData(fp, mIOPtrs.sliceClusters, mIOPtrs.nSliceClusters, mIOMem.sliceClusters, InOutPointerType::SLICE_OUT_CLUSTER);
  ReadData(fp, &mIOPtrs.mcLabelsTPC, &mIOPtrs.nMCLabelsTPC, &mIOMem.mcLabelsTPC, InOutPointerType::MC_LABEL_TPC);
  ReadData(fp, &mIOPtrs.mcInfosTPC, &mIOPtrs.nMCInfosTPC, &mIOMem.mcInfosTPC, InOutPointerType::MC_INFO_TPC);
  ReadData(fp, &mIOPtrs.mergedTracks, &mIOPtrs.nMergedTracks, &mIOMem.mergedTracks, InOutPointerType::MERGED_TRACK);
  ReadData(fp, &mIOPtrs.mergedTrackHits, &mIOPtrs.nMergedTrackHits, &mIOMem.mergedTrackHits, InOutPointerType::MERGED_TRACK_HIT);
  ReadData(fp, &mIOPtrs.trdTracks, &mIOPtrs.nTRDTracks, &mIOMem.trdTracks, InOutPointerType::TRD_TRACK);
  ReadData(fp, &mIOPtrs.trdTracklets, &mIOPtrs.nTRDTracklets, &mIOMem.trdTracklets, InOutPointerType::TRD_TRACKLET);
  ReadData(fp, &mIOPtrs.trdTrackletsMC, &mIOPtrs.nTRDTrackletsMC, &mIOMem.trdTrackletsMC, InOutPointerType::TRD_TRACKLET_MC);
  fclose(fp);
  (void)r;
  for (unsigned int i = 0; i < NSLICES; i++) {
    for (unsigned int j = 0; j < mIOPtrs.nClusterData[i]; j++) {
      ptrClusterData[i][j].id = nClustersTotal++;
      if ((unsigned int)ptrClusterData[i][j].amp >= 25 * 1024) {
        GPUError("Invalid cluster charge, truncating (%d >= %d)", (int)ptrClusterData[i][j].amp, 25 * 1024);
        ptrClusterData[i][j].amp = 25 * 1024 - 1;
      }
    }
    for (unsigned int j = 0; j < mIOPtrs.nRawClusters[i]; j++) {
      if ((unsigned int)mIOMem.rawClusters[i][j].GetCharge() >= 25 * 1024) {
        GPUError("Invalid raw cluster charge, truncating (%d >= %d)", (int)ptrRawClusters[i][j].GetCharge(), 25 * 1024);
        ptrRawClusters[i][j].SetCharge(25 * 1024 - 1);
      }
      if ((unsigned int)mIOPtrs.rawClusters[i][j].GetQMax() >= 1024) {
        GPUError("Invalid raw cluster charge max, truncating (%d >= %d)", (int)ptrRawClusters[i][j].GetQMax(), 1024);
        ptrRawClusters[i][j].SetQMax(1024 - 1);
      }
    }
  }

  return (0);
}

void GPUChainTracking::DumpSettings(const char* dir)
{
  std::string f;
  if (processors()->calibObjects.fastTransform != nullptr) {
    f = dir;
    f += "tpctransform.dump";
    DumpFlatObjectToFile(processors()->calibObjects.fastTransform, f.c_str());
  }
  if (processors()->calibObjects.tpcPadGain != nullptr) {
    f = dir;
    f += "tpcpadgaincalib.dump";
    DumpStructToFile(processors()->calibObjects.tpcPadGain, f.c_str());
  }
#ifdef HAVE_O2HEADERS
  if (processors()->calibObjects.dEdxSplines != nullptr) {
    f = dir;
    f += "dedxsplines.dump";
    DumpFlatObjectToFile(processors()->calibObjects.dEdxSplines, f.c_str());
  }
  if (processors()->calibObjects.matLUT != nullptr) {
    f = dir;
    f += "matlut.dump";
    DumpFlatObjectToFile(processors()->calibObjects.matLUT, f.c_str());
  }
  if (processors()->calibObjects.trdGeometry != nullptr) {
    f = dir;
    f += "trdgeometry.dump";
    DumpStructToFile(processors()->calibObjects.trdGeometry, f.c_str());
  }

#endif
}

void GPUChainTracking::ReadSettings(const char* dir)
{
  std::string f;
  f = dir;
  f += "tpctransform.dump";
  mTPCFastTransformU = ReadFlatObjectFromFile<TPCFastTransform>(f.c_str());
  processors()->calibObjects.fastTransform = mTPCFastTransformU.get();
  f = dir;
  f += "tpcpadgaincalib.dump";
  mTPCPadGainCalibU = ReadStructFromFile<TPCPadGainCalib>(f.c_str());
  processors()->calibObjects.tpcPadGain = mTPCPadGainCalibU.get();
#ifdef HAVE_O2HEADERS
  f = dir;
  f += "dedxsplines.dump";
  mdEdxSplinesU = ReadFlatObjectFromFile<TPCdEdxCalibrationSplines>(f.c_str());
  processors()->calibObjects.dEdxSplines = mdEdxSplinesU.get();
  f = dir;
  f += "matlut.dump";
  mMatLUTU = ReadFlatObjectFromFile<o2::base::MatLayerCylSet>(f.c_str());
  processors()->calibObjects.matLUT = mMatLUTU.get();
  f = dir;
  f += "trdgeometry.dump";
  mTRDGeometryU = ReadStructFromFile<o2::trd::GeometryFlat>(f.c_str());
  processors()->calibObjects.trdGeometry = mTRDGeometryU.get();
#endif
}
