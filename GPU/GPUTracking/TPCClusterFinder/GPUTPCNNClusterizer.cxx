// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNNClusterizer.cxx
/// \author Christian Sonnabend

#include "GPUTPCNNClusterizer.h"

#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"
#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#include "MCLabelAccumulator.h"
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, char onlyMC)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(
    MCLabelAccumulator labelAcc(clusterer));

  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;

  GPUTPCNNClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, smem, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
}

void GPUTPCNNClusterizer::exec(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, char onlyMC)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(
    MCLabelAccumulator labelAcc(clusterer));

  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;

  OnnxModel model_class, model_reg;
  std::string path_class = "", path_reg = "";

  model_class.init(path_class, 1, 0);
  model_reg.init(path_reg, 1, 0);

  GPUTPCNNClusterizer::nn_clusterizer(model_class, model_reg, clusterer, clusterer.mPmemory->fragment, smem, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow, 3, 3, 3, 1, 0.16, 1);
}

int GPUTPCNNClusterizer::padOffset(int row_ref, int row_current)
{
  return (int)((pad_row_max[row_ref] - pad_row_max[row_current]) / 2);
}

// ---------------------------------
bool GPUTPCNNClusterizer::isBoundary(int row, int pad, int global_shift)
{
  if (row < 0 || pad < 0) {
    return true;
  } else if (row <= 62) {
    if (pad < (pad_row_max[o2::tpc::constants::MAXGLOBALPADROW-1] - pad_row_max[row]) / 2 || pad > (pad_row_max[o2::tpc::constants::MAXGLOBALPADROW-1] + pad_row_max[row]) / 2) {
      return true;
    } else {
      return false;
    }
  } else if (row <= 62 + global_shift) {
    return true;
  } else if (row <= o2::tpc::constants::MAXGLOBALPADROW-1 + global_shift) {
    if (pad < (pad_row_max[o2::tpc::constants::MAXGLOBALPADROW-1] - pad_row_max[row - global_shift]) / 2 || pad > (pad_row_max[o2::tpc::constants::MAXGLOBALPADROW-1] + pad_row_max[row - global_shift]) / 2) {
      return true;
    } else {
      return false;
    }
  } else if (row > o2::tpc::constants::MAXGLOBALPADROW-1 + global_shift) {
    return true;
  } else {
    return false;
  }
}

void GPUTPCNNClusterizer::nn_clusterizer(OnnxModel model_class, OnnxModel model_reg, 
                                          processorType& clusterer,
                                          const CfFragment& fragment,
                                          GPUSharedMemory& smem,
                                          const Array2D<PackedCharge>& chargeMap,
                                          const ChargePos* filteredPeakPositions,
                                          const GPUSettingsRec& calib,
                                          MCLabelAccumulator* labelAcc,
                                          uint clusternum,
                                          uint maxClusterPerRow,
                                          uint* clusterInRow,
                                          tpc::ClusterNative* clusterByRow,
                                          uint* clusterPosInRow
                                          int in_row, int in_pad, int in_time, bool add_index_data, float class_threshold, bool sigmoid_transform){

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  std::vector<float> input_data(clusterer.mPmemory->counters.nClusters * (2*in_row + 1) * (2*in_pad + 1) * (2*in_time + 1));
  float classification_threshold = class_threshold;
  if(sigmoid_transform){
    classification_threshold = (float)std::log(class_threshold/(1.f-class_threshold));
  }
  
  for(float cls = 0; cls < clusterer.mPmemory->counters.nClusters; cls++){
    ChargePos peak = clusterer.mPfilteredPeakPositions[cls];
    int row = peak.row(), pad = peak.pad(), time = peak.time();
    float central_charge = chargeMap[peak].unpack();
    unsigned int glo_idx = cls * ((2*in_row + 1) + (2*in_pad + 1) * (2*in_time + 1));
    for(int r = -in_row; r <= in_row; r++){
      for(int p = -in_pad; p <= in_pad; p++){
        for(int t = -in_time; t <= in_time; t++){
          int offset = padOffset(row, row + r);
          if(isBoundary(row + r, pad + p + offset)){
            continue;
          } else {
            unsigned int idx = glo_idx + (row + r) * (2*in_pad + 1) * (2*in_time + 1) + (pad + p) * (2*in_time + 1) + (time + t);
            ChargePos tmp_pos(row + r, pad + p + offset, time + t);
            input_data[idx] = (chargeMap[tmp_pos].unpack() / central_charge);
          }
        }
      }
    }
  }
  std::vector<float> out_class = model_class.inference_vector(input_data, clusterer.mPmemory->counters.nClusters);
  std::vector<float> out_reg = model_reg.inference_vector(input_data, clusterer.mPmemory->counters.nClusters);
  int num_outputs = model_reg.getNumOutputNodes()[0][1];

  for(int cls = 0; cls < clusterer.mPmemory->counters.nClusters; cls++){
    if(out_class > classification_threshold){
      int idx = cls * num_outputs;
      ChargePos peak = clusterer.mPfilteredPeakPositions[cls];
      ClusterAccumulator pc;
      pc.setFull(chargeMap[peak].unpack() * out_reg[idx + 4], peak.pad() + out_reg[idx], out_reg[idx + 2], peak.time() + out_reg[idx + 1], out_reg[idx + 3], 0, 0);
      tpc::ClusterNative myCluster;
      bool rejectCluster = !pc.toNative(pos, chargeMap[peak].unpack(), myCluster, clusterer.Param());
      if (rejectCluster) {
        if (clusterPosInRow) {
          clusterPosInRow[idx] = maxClusterPerRow;
        }
        return;
      }

      uint rowIndex = 0;
      if (clusterByRow != nullptr) {
        rowIndex = sortIntoBuckets(
          clusterer,
          myCluster,
          pos.row(),
          maxClusterPerRow,
          clusterInRow,
          clusterByRow);
        if (clusterPosInRow != nullptr) {
          clusterPosInRow[idx] = rowIndex;
        }
      } else if (clusterPosInRow) {
        rowIndex = clusterPosInRow[idx];
      }

      CPU_ONLY(labelAcc->commit(pos.row(), rowIndex, maxClusterPerRow));
    }
  }

}



GPUdii() void GPUTPCNNClusterizer::computeClustersImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                       processorType& clusterer,
                                                       const CfFragment& fragment,
                                                       GPUSharedMemory& smem,
                                                       const Array2D<PackedCharge>& chargeMap,
                                                       const ChargePos* filteredPeakPositions,
                                                       const GPUSettingsRec& calib,
                                                       MCLabelAccumulator* labelAcc,
                                                       uint clusternum,
                                                       uint maxClusterPerRow,
                                                       uint* clusterInRow,
                                                       tpc::ClusterNative* clusterByRow,
                                                       uint* clusterPosInRow)
{
  uint idx = get_global_id(0);

  // For certain configurations dummy work items are added, so the total
  // number of work items is dividable by 64.
  // These dummy items also compute the last cluster but discard the result.
  ChargePos pos = filteredPeakPositions[CAMath::Min(idx, clusternum - 1)];
  Charge charge = chargeMap[pos].unpack();

  ClusterAccumulator pc;
  CPU_ONLY(labelAcc->collect(pos, charge));

  buildCluster(
    calib,
    chargeMap,
    pos,
    smem.posBcast,
    smem.buf,
    smem.innerAboveThreshold,
    &pc,
    labelAcc);

  if (idx >= clusternum) {
    return;
  }
  if (fragment.isOverlap(pos.time())) {
    if (clusterPosInRow) {
      clusterPosInRow[idx] = maxClusterPerRow;
    }
    return;
  }
  pc.finalize(pos, charge, fragment.start, clusterer.Param().tpcGeometry);

  tpc::ClusterNative myCluster;
  bool rejectCluster = !pc.toNative(pos, charge, myCluster, clusterer.Param());

  if (rejectCluster) {
    if (clusterPosInRow) {
      clusterPosInRow[idx] = maxClusterPerRow;
    }
    return;
  }

  uint rowIndex = 0;
  if (clusterByRow != nullptr) {
    rowIndex = sortIntoBuckets(
      clusterer,
      myCluster,
      pos.row(),
      maxClusterPerRow,
      clusterInRow,
      clusterByRow);
    if (clusterPosInRow != nullptr) {
      clusterPosInRow[idx] = rowIndex;
    }
  } else if (clusterPosInRow) {
    rowIndex = clusterPosInRow[idx];
  }

  CPU_ONLY(labelAcc->commit(pos.row(), rowIndex, maxClusterPerRow));
}

GPUdii() void GPUTPCNNClusterizer::updateClusterInner(
  const GPUSettingsRec& calib,
  ushort lid,
  ushort N,
  const PackedCharge* buf,
  const ChargePos& pos,
  ClusterAccumulator* cluster,
  MCLabelAccumulator* labelAcc,
  uchar* innerAboveThreshold)
{
  uchar aboveThreshold = 0;

  GPUCA_UNROLL(U(), U())
  for (ushort i = 0; i < N; i++) {
    Delta2 d = cfconsts::InnerNeighbors[i];

    PackedCharge p = buf[N * lid + i];

    Charge q = cluster->updateInner(p, d);

    CPU_ONLY(
      labelAcc->collect(pos.delta(d), q));

    aboveThreshold |= (uchar(q > calib.tpc.cfInnerThreshold) << i);
  }

  innerAboveThreshold[lid] = aboveThreshold;

  GPUbarrier();
}

GPUdii() void GPUTPCNNClusterizer::updateClusterOuter(
  ushort lid,
  ushort N,
  ushort M,
  ushort offset,
  const PackedCharge* buf,
  const ChargePos& pos,
  ClusterAccumulator* cluster,
  MCLabelAccumulator* labelAcc)
{
  GPUCA_UNROLL(U(), U())
  for (ushort i = offset; i < M + offset; i++) {
    PackedCharge p = buf[N * lid + i];

    Delta2 d = cfconsts::OuterNeighbors[i];

    Charge q = cluster->updateOuter(p, d);
    static_cast<void>(q); // Avoid unused varible warning on GPU.

    CPU_ONLY(
      labelAcc->collect(pos.delta(d), q));
  }
}

GPUdii() void GPUTPCNNClusterizer::buildCluster(
  const GPUSettingsRec& calib,
  const Array2D<PackedCharge>& chargeMap,
  ChargePos pos,
  ChargePos* posBcast,
  PackedCharge* buf,
  uchar* innerAboveThreshold,
  ClusterAccumulator* myCluster,
  MCLabelAccumulator* labelAcc)
{
  ushort ll = get_local_id(0);

  posBcast[ll] = pos;
  GPUbarrier();

  CfUtils::blockLoad<PackedCharge>(
    chargeMap,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    8,
    cfconsts::InnerNeighbors,
    posBcast,
    buf);
  updateClusterInner(
    calib,
    ll,
    8,
    buf,
    pos,
    myCluster,
    labelAcc,
    innerAboveThreshold);

  ushort wgSizeHalf = (SCRATCH_PAD_WORK_GROUP_SIZE + 1) / 2;

  bool inGroup1 = ll < wgSizeHalf;

  ushort llhalf = (inGroup1) ? ll : (ll - wgSizeHalf);

  CfUtils::condBlockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    cfconsts::OuterNeighbors,
    posBcast,
    innerAboveThreshold,
    buf);

  if (inGroup1) {
    updateClusterOuter(
      llhalf,
      16,
      16,
      0,
      buf,
      pos,
      myCluster,
      labelAcc);
  }

#if defined(GPUCA_GPUCODE)
  CfUtils::condBlockLoad(
    chargeMap,
    wgSizeHalf,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    cfconsts::OuterNeighbors,
    posBcast + wgSizeHalf,
    innerAboveThreshold + wgSizeHalf,
    buf);
  if (!inGroup1) {
    updateClusterOuter(
      llhalf,
      16,
      16,
      0,
      buf,
      pos,
      myCluster,
      labelAcc);
  }
#endif
}

GPUd() uint GPUTPCNNClusterizer::sortIntoBuckets(processorType& clusterer, const tpc::ClusterNative& cluster, uint row, uint maxElemsPerBucket, uint* elemsInBucket, tpc::ClusterNative* buckets)
{
  uint index = CAMath::AtomicAdd(&elemsInBucket[row], 1u);
  if (index < maxElemsPerBucket) {
    buckets[maxElemsPerBucket * row + index] = cluster;
  } else {
    clusterer.raiseError(GPUErrors::ERROR_CF_ROW_CLUSTER_OVERFLOW, clusterer.mISlice * 1000 + row, index, maxElemsPerBucket);
    CAMath::AtomicExch(&elemsInBucket[row], maxElemsPerBucket);
  }
  return index;
}
