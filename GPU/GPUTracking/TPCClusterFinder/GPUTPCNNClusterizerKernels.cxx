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

/// \file GPUTPCNNClusterizerKernels.cxx
/// \author Christian Sonnabend

#include "clusterFinderDefs.h"
#include "PackedCharge.h"
#include "GPUTPCNNClusterizerKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "GPUTPCCFClusterizer.h"
#include "GPUTPCGeometry.h"

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"
#include "ML/3rdparty/GPUORTFloat16.h"

#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#include "MCLabelAccumulator.h"
#endif

#ifdef GPUCA_GPUCODE
#include "GPUTPCCFClusterizer.inc"
#endif

// Defining individual thread functions for data filling, determining the class label and running the CF clusterizer
template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::runCfClusterizer>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAcc(clusterer));
  tpc::ClusterNative* clusterOut = clusterer.mPclusterByRow;
  int8_t isAccepted = (clustererNN.mNnClusterizerUseClassification ? (clustererNN.mOutputDataClass[CAMath::Min(glo_idx, (uint32_t)clusterer.mPmemory->counters.nClusters - 1)] > 0) : 1);
  GPUTPCCFClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, reinterpret_cast<GPUTPCCFClusterizer::GPUSharedMemory&>(smem), chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow, isAccepted);
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::fillInputNNCPU>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];

  uint32_t glo_idx = get_global_id(0);
  if (glo_idx + batchStart >= clusterer.mPmemory->counters.nClusters || glo_idx >= (uint32_t)clustererNN.mNnClusterizerBatchedMode) {
    return;
  }

  uint32_t write_idx = glo_idx * clustererNN.mNnClusterizerElementSize;

  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfArray2D<uint8_t> isPeakMap(clusterer.mPpeakMap);
  CfChargePos peak = clusterer.mPfilteredPeakPositions[CAMath::Min(glo_idx + batchStart, (uint32_t)(clusterer.mPmemory->counters.nClusters - 1))];
  int32_t row = static_cast<int>(peak.row());
  int32_t pad = static_cast<int>(peak.pad());
  int32_t time = static_cast<int>(peak.time());
  float central_charge = static_cast<float>(chargeMap[peak].unpack());
  int32_t row_offset = GPUTPCNNClusterizerKernels::rowOffset(row, clustererNN.mNnClusterizerSizeInputRow);

  for (int32_t r = -clustererNN.mNnClusterizerSizeInputRow; r <= clustererNN.mNnClusterizerSizeInputRow; ++r) {
    int32_t target_row = row + r;
    bool is_row_boundary = (target_row < 0) || (target_row >= o2::tpc::constants::MAXGLOBALPADROW);
    int32_t pad_offset = is_row_boundary ? 0 : GPUTPCNNClusterizerKernels::padOffset(row, target_row);

    for (int32_t p = -clustererNN.mNnClusterizerSizeInputPad + pad_offset; p <= clustererNN.mNnClusterizerSizeInputPad + pad_offset; ++p) {
      int32_t target_pad = pad + p;
      bool is_boundary = is_row_boundary || GPUTPCNNClusterizerKernels::isBoundary(target_row + row_offset, target_pad, clustererNN.mNnClusterizerSizeInputRow);

      for (int32_t t = -clustererNN.mNnClusterizerSizeInputTime; t <= clustererNN.mNnClusterizerSizeInputTime; ++t) {
        int32_t target_time = time + t;

        if (is_boundary || target_time < 0 || target_time >= clustererNN.maxAllowedTimebin) {
          // Fill boundary value
          float boundary_value = static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue);
          if (dtype == 0) {
            clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)boundary_value;
          } else {
            clustererNN.mInputData_32[write_idx] = boundary_value;
          }
        } else {
          CfChargePos tmp_pos(target_row, target_pad, target_time);
          float normalized_charge = static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge;
          if (dtype == 0) {
            clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)normalized_charge;
          } else {
            clustererNN.mInputData_32[write_idx] = normalized_charge;
          }
        }
        // if((CAMath::Abs(static_cast<float>(clustererNN.mInputData_16_Test[write_idx]) - static_cast<float>(clustererNN.mInputData_16[write_idx])) > 1e-4) && ((glo_idx + batchStart) < clusterer.mPmemory->counters.nClusters)) {
        //   printf("Warning: Input data mismatch at index %d, %d - row, pad, time: %d, %d, %d : %f -> %f\n", glo_idx, glo_idx + batchStart, r, p, t,
        //          static_cast<float>(clustererNN.mInputData_16_Test[write_idx]), static_cast<float>(clustererNN.mInputData_16[write_idx]));
        // }
        write_idx++;
      }
    }
  }

  if (clustererNN.mNnClusterizerAddIndexData) {
    if (dtype == 0) {
      clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(sector) / o2::tpc::constants::MAXSECTOR);
      clustererNN.mInputData_16[write_idx + 1] = (OrtDataType::Float16_t)(static_cast<float>(row) / o2::tpc::constants::MAXGLOBALPADROW);
      clustererNN.mInputData_16[write_idx + 2] = (OrtDataType::Float16_t)(static_cast<float>(pad) / GPUTPCGeometry::NPads(row));
    } else {
      clustererNN.mInputData_32[write_idx] = static_cast<float>(sector) / o2::tpc::constants::MAXSECTOR;
      clustererNN.mInputData_32[write_idx + 1] = static_cast<float>(row) / o2::tpc::constants::MAXGLOBALPADROW;
      clustererNN.mInputData_32[write_idx + 2] = static_cast<float>(pad) / GPUTPCGeometry::NPads(row);
    }
  }

  if (!clustererNN.mNnClusterizerSetDeconvolutionFlags) {
    clustererNN.mClusterFlags[2 * glo_idx] = 0;
    clustererNN.mClusterFlags[2 * glo_idx + 1] = 0;

    for (uint16_t i = 0; i < 8; ++i) {
      Delta2 d = cfconsts::InnerNeighbors[i];
      CfChargePos tmp_pos = peak.delta(d);
      clustererNN.mClusterFlags[2 * glo_idx] += CfUtils::isPeak(isPeakMap[tmp_pos]);
    }
    clustererNN.mClusterFlags[2 * glo_idx + 1] = clustererNN.mClusterFlags[2 * glo_idx];
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::fillInputNNGPU>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];

  if (glo_idx >= (uint32_t)clustererNN.mNnClusterizerBatchedMode * clustererNN.mNnClusterizerRowTimeSizeFull) {
    return;
  }

  uint32_t base_idx = glo_idx / clustererNN.mNnClusterizerRowTimeSizeFull;
  uint32_t transient_index = glo_idx - (base_idx * clustererNN.mNnClusterizerRowTimeSizeFull);

  // Early exit for out-of-bounds threads
  if (base_idx + batchStart >= clusterer.mPmemory->counters.nClusters) {
    return;
  }
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfArray2D<uint8_t> isPeakMap(clusterer.mPpeakMap);

  // Use dedicated neural network shared memory arrays for warp-level caching
  // First thread in each warp loads shared data
  CfChargePos peak = clusterer.mPfilteredPeakPositions[CAMath::Min(base_idx + batchStart, (uint32_t)(clusterer.mPmemory->counters.nClusters - 1))];
  float central_charge = static_cast<float>(chargeMap[peak].unpack());
  int32_t row = static_cast<int>(peak.row());
  int32_t pad = static_cast<int>(peak.pad());
  int32_t time = static_cast<int>(peak.time());

  // Handle index data with fewer branches
  if (clustererNN.mNnClusterizerAddIndexData && transient_index >= clustererNN.mNnClusterizerRowTimeSize) {
    int32_t data_idx = transient_index - clustererNN.mNnClusterizerRowTimeSize;
    uint32_t write_idx = base_idx * clustererNN.mNnClusterizerElementSize + clustererNN.mNnClusterizerChargeArraySize + data_idx;

    float index_values[3] = {
      static_cast<float>(sector) / o2::tpc::constants::MAXSECTOR,
      static_cast<float>(row) / o2::tpc::constants::MAXGLOBALPADROW,
      static_cast<float>(pad) / GPUTPCGeometry::NPads(row)};

    if (dtype == 0) {
      clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)index_values[data_idx];
    } else {
      clustererNN.mInputData_32[write_idx] = index_values[data_idx];
    }

    // Handle deconvolution flags only once per cluster (last thread in element)
    if (!clustererNN.mNnClusterizerSetDeconvolutionFlags && data_idx == 2) {
      uint8_t cluster_flags = 0;
      for (uint16_t i = 0; i < 8; i++) {
        Delta2 d = cfconsts::InnerNeighbors[i];
        CfChargePos tmp_pos = peak.delta(d);
        cluster_flags += CfUtils::isPeak(isPeakMap[tmp_pos]);
      }
      clustererNN.mClusterFlags[2 * base_idx] = cluster_flags;
      clustererNN.mClusterFlags[2 * base_idx + 1] = cluster_flags;
    }
    return;
  }

  // Main data processing - optimize index calculations
  if (transient_index < clustererNN.mNnClusterizerRowTimeSize) {
    // Optimize 3D index calculation
    int32_t row_idx = transient_index / clustererNN.mNnClusterizerFullTimeSize;
    int32_t r_local = row_idx - clustererNN.mNnClusterizerSizeInputRow;
    int32_t time_idx = transient_index - row_idx * clustererNN.mNnClusterizerFullTimeSize;
    int32_t t_local = time_idx - clustererNN.mNnClusterizerSizeInputTime;
    int32_t write_idx = base_idx * clustererNN.mNnClusterizerElementSize + row_idx * clustererNN.mNnClusterizerPadTimeSize + time_idx;

    // Early boundary check for row
    int32_t target_row = row + r_local;
    int8_t is_row_boundary = (target_row < 0) || (target_row > (o2::tpc::constants::MAXGLOBALPADROW - 1));

    // Calculate offsets
    int32_t row_offset = GPUTPCNNClusterizerKernels::rowOffset(row, clustererNN.mNnClusterizerSizeInputRow);
    int32_t pad_offset = GPUTPCNNClusterizerKernels::padOffset(row, target_row);
    for (int32_t p_local = -clustererNN.mNnClusterizerSizeInputPad + pad_offset; p_local <= clustererNN.mNnClusterizerSizeInputPad + pad_offset; p_local++) {
      if (is_row_boundary) {
        // Use boundary fill value
        float boundary_val = static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue);
        if (dtype == 0) {
          clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)boundary_val;
        } else {
          clustererNN.mInputData_32[write_idx] = boundary_val;
        }
        write_idx += clustererNN.mNnClusterizerFullTimeSize; // Move to next pad position
        continue;
      }

      // Calculate target pad and time
      int32_t target_pad = pad + p_local;
      int32_t target_time = time + t_local;

      // Optimized boundary check
      int8_t is_boundary = GPUTPCNNClusterizerKernels::isBoundary(target_row + row_offset, target_pad, clustererNN.mNnClusterizerSizeInputRow) || (target_time < 0) || (target_time >= clustererNN.maxAllowedTimebin);

      float output_value;
      if (is_boundary) {
        output_value = static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue);
      } else {
        // Coalesced memory access - create position and read charge
        CfChargePos tmp_pos(target_row, target_pad, target_time);
        output_value = static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge; // Normalize by central charge
      }

      // Write output with reduced branching
      if (dtype == 0) {
        clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)output_value;
      } else {
        clustererNN.mInputData_32[write_idx] = output_value;
      }
      // if (write_idx >= clustererNN.mNnClusterizerElementSize * clustererNN.mNnClusterizerBatchedMode) {
      //   printf("Error: Write index out of bounds (central array)! %d >= %d (write_idx: %d, base_idx: %d, transient_index: %d, row_idx: %d, time_idx: %d, r_local: %d, t_local: %d)\n",
      //          write_idx, (int)(clustererNN.mNnClusterizerElementSize * clustererNN.mNnClusterizerBatchedMode), write_idx, base_idx, transient_index, row_idx, time_idx, r_local, t_local);
      // }
      // if ((clusterer.mPmemory->counters.nClusters - batchStart) < clustererNN.mNnClusterizerBatchedMode) {
      //   if (write_idx >= ((clusterer.mPmemory->counters.nClusters - batchStart) * clustererNN.mNnClusterizerElementSize)) {
      //     printf("Error: Write index out of bounds (end of array)! %d >= %d (write_idx: %d, base_idx: %d, transient_index: %d, row_idx: %d, time_idx: %d, r_local: %d, t_local: %d)\n",
      //           write_idx, (int)((clusterer.mPmemory->counters.nClusters - batchStart) * clustererNN.mNnClusterizerElementSize), write_idx, base_idx, transient_index, row_idx, time_idx, r_local, t_local);
      //   }
      //   if (write_idx > ((clusterer.mPmemory->counters.nClusters - batchStart) * clustererNN.mNnClusterizerElementSize - 5)) {
      //     printf("Sanity check (should appear only once) %d == %d (write_idx: %d, base_idx: %d, transient_index: %d, row_idx: %d, time_idx: %d, r_local: %d, t_local: %d)\n",
      //           write_idx, (int)((clusterer.mPmemory->counters.nClusters - batchStart) * clustererNN.mNnClusterizerElementSize - 4), write_idx, base_idx, transient_index, row_idx, time_idx, r_local, t_local);
      //   }
      // }

      write_idx += clustererNN.mNnClusterizerFullTimeSize; // Move to next pad position
    }
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::determineClass1Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  if (glo_idx + batchStart >= clusterer.mPmemory->counters.nClusters || glo_idx >= (uint32_t)clustererNN.mNnClusterizerBatchedMode) {
    return;
  }
  if (clustererNN.mNnClusterizerUseClassification) {
    if (dtype == 0) {
      clustererNN.mOutputDataClass[glo_idx + batchStart] = (int32_t)((clustererNN.mModelProbabilities_16[glo_idx]).ToFloat() > clustererNN.mNnClassThreshold);
    } else if (dtype == 1) {
      clustererNN.mOutputDataClass[glo_idx + batchStart] = (int32_t)(clustererNN.mModelProbabilities_32[glo_idx] > clustererNN.mNnClassThreshold);
    }
  } else {
    clustererNN.mOutputDataClass[glo_idx + batchStart] = 1;
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::determineClass2Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  if (glo_idx + batchStart >= clusterer.mPmemory->counters.nClusters || glo_idx >= (uint32_t)clustererNN.mNnClusterizerBatchedMode) {
    return;
  }
  if (clustererNN.mNnClusterizerUseClassification) {
    uint32_t elem_iterator = glo_idx * clustererNN.mNnClusterizerModelClassNumOutputNodes;
    float current_max_prob = 0.f; // If the neural network doesn't contain the softmax as a last layer, the outputs can range in [-infty, infty]
    uint32_t class_label = 0;
    for (uint32_t pIdx = elem_iterator; pIdx < elem_iterator + clustererNN.mNnClusterizerModelClassNumOutputNodes; pIdx++) {
      if (pIdx == elem_iterator) {
        if (dtype == 0) {
          current_max_prob = static_cast<float>(clustererNN.mModelProbabilities_16[pIdx]);
        } else if (dtype == 1) {
          current_max_prob = clustererNN.mModelProbabilities_32[pIdx];
        }
      } else {
        if (dtype == 0) {
          current_max_prob = CAMath::Max(current_max_prob, clustererNN.mModelProbabilities_16[pIdx].ToFloat());
        } else if (dtype == 1) {
          current_max_prob = CAMath::Max(current_max_prob, clustererNN.mModelProbabilities_32[pIdx]);
        }
      }
    }
    // uint32_t class_label = std::distance(elem_iterator, std::max_element(elem_iterator, elem_iterator + clustererNN.mNnClusterizerModelClassNumOutputNodes)); // Multiple outputs of the class network are the probabilities for each class. The highest one "wins"
    clustererNN.mOutputDataClass[glo_idx + batchStart] = class_label;
    if (class_label > 1) {
      clustererNN.mClusterFlags[2 * glo_idx] = 1;
      clustererNN.mClusterFlags[2 * glo_idx + 1] = 1;
    }
  } else {
    clustererNN.mOutputDataClass[glo_idx + batchStart] = 1;
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishClass1Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  if (glo_idx >= (uint32_t)clustererNN.mNnClusterizerBatchedMode) {
    return;
  }

  uint32_t maxClusterNum = clusterer.mPmemory->counters.nClusters;
  uint32_t full_glo_idx = glo_idx + batchStart;
  int32_t model_output_index = glo_idx * clustererNN.mNnClusterizerModelReg1NumOutputNodes;

  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  uint32_t peakIndex = CAMath::Min(full_glo_idx, maxClusterNum - 1);
  CfChargePos peak = clusterer.mPfilteredPeakPositions[peakIndex];
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem);

  if (full_glo_idx >= maxClusterNum) {
    if (withMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect(peak, central_charge));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        peak,
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }
    return;
  }

  tpc::ClusterNative* clusterOut = clusterer.mPclusterByRow;

  ClusterAccumulator pc;

  if (withMC) {
    ClusterAccumulator dummy_pc;
    CPU_ONLY(labelAcc->collect(peak, central_charge));
    GPUTPCCFClusterizer::buildCluster(
      clusterer.Param().rec,
      chargeMap,
      peak,
      smem.posBcast,
      smem.buf,
      smem.innerAboveThreshold,
      &dummy_pc,
      labelAcc);
  }
  if ((clusterer.mPmemory->fragment).isOverlap(peak.time())) {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }

  bool notSinglePad = false, notSingleTime = false;
  for (uint16_t i = 0; i < 8; i++) {
    Delta2 d = cfconsts::InnerNeighbors[i];
    CfChargePos tmp_pos = peak.delta(d);
    float v = static_cast<float>(chargeMap[tmp_pos].unpack());
    notSinglePad |= (d.x != 0) && (v > 0.f);
    notSingleTime |= (d.y != 0) && (v > 0.f);
  }

  float publishPadPosition = 0.f, publishTimePosition = 0.f;
  if (dtype == 0) {
    publishPadPosition = static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg1_16[model_output_index].ToFloat();
    publishTimePosition = static_cast<float>(peak.time()) + clustererNN.mOutputDataReg1_16[model_output_index + 1].ToFloat();
    isBoundaryPublish(full_glo_idx, static_cast<int32_t>(peak.row()), publishPadPosition, publishTimePosition);
    pc.setFull(central_charge * clustererNN.mOutputDataReg1_16[model_output_index + 4].ToFloat(),
               publishPadPosition,
               notSinglePad ? clustererNN.mOutputDataReg1_16[model_output_index + 2].ToFloat() : 0.f,
               (clusterer.mPmemory->fragment).start + publishTimePosition,
               notSingleTime ? clustererNN.mOutputDataReg1_16[model_output_index + 3].ToFloat() : 0.f,
               clustererNN.mClusterFlags[2 * glo_idx],
               clustererNN.mClusterFlags[2 * glo_idx + 1]);
  } else {
    publishPadPosition = static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg1_32[model_output_index];
    publishTimePosition = static_cast<float>(peak.time()) + clustererNN.mOutputDataReg1_32[model_output_index + 1];
    isBoundaryPublish(full_glo_idx, static_cast<int32_t>(peak.row()), publishPadPosition, publishTimePosition);
    pc.setFull(central_charge * clustererNN.mOutputDataReg1_32[model_output_index + 4],
               publishPadPosition,
               notSinglePad ? clustererNN.mOutputDataReg1_32[model_output_index + 2] : 0.f,
               (clusterer.mPmemory->fragment).start + publishTimePosition,
               notSingleTime ? clustererNN.mOutputDataReg1_32[model_output_index + 3] : 0.f,
               clustererNN.mClusterFlags[2 * glo_idx],
               clustererNN.mClusterFlags[2 * glo_idx + 1]);
  }

  // if (boundaryFlag != 0) { // Prints the entire NN input for the given index
  //   // Build a simple buffer manually (float with 3 decimals)
  //   const int MAX_CHARS = 4096;
  //   char buffer[MAX_CHARS];
  //   int pos = 0;
  //
  //   auto appendChar = [&](char c) {
  //     if (pos < MAX_CHARS - 1) buffer[pos++] = c;
  //   };
  //   auto appendStr = [&](const char* s) {
  //     while (*s && pos < MAX_CHARS - 1) buffer[pos++] = *s++;
  //   };
  //   auto appendUInt = [&](uint32_t v) {
  //     char tmp[16]; int tp = 0;
  //     if (v == 0) { appendChar('0'); return; }
  //     while (v && tp < 16) { tmp[tp++] = char('0' + (v % 10)); v /= 10; }
  //     while (tp--) appendChar(tmp[tp]);
  //   };
  //   auto appendInt = [&](int v) {
  //     if (v < 0) { appendChar('-'); v = -v; }
  //     appendUInt((uint32_t)v);
  //   };
  //   auto appendFloat = [&](float f) {
  //     if (f < 0) { appendChar('-'); f = -f; }
  //     int ip = (int)f;
  //     float frac = f - (float)ip;
  //     appendInt(ip);
  //     appendChar('.');
  //     for (int i = 0; i < 3; i++) {
  //       frac *= 10.f;
  //       int d = (int)frac;
  //       appendChar((char)('0' + (d < 0 ? 0 : (d > 9 ? 9 : d))));
  //       frac -= d;
  //       if (frac < 0) frac = 0;
  //     }
  //   };
  //
  //   appendStr("(NN CLUS) DEBUG: Boundary cluster detected (sector ");
  //   appendUInt(sector);
  //   appendStr(", row ");
  //   appendUInt(peak.row());
  //   appendStr(", pad ");
  //   appendFloat(publishPadPosition);
  //   appendStr(", time ");
  //   appendFloat(publishTimePosition);
  //   appendStr(") [glo_idx=");
  //   appendUInt(glo_idx);
  //   appendStr(" elemSize=");
  //   appendInt(clustererNN.mNnClusterizerElementSize);
  //   appendStr(" dtype=");
  //   appendInt(dtype);
  //   appendStr("] INPUT:");
  //
  //   int elemSize = clustererNN.mNnClusterizerElementSize;
  //   int baseIdx = glo_idx * elemSize;
  //
  //   int maxPrint = elemSize;
  //   for (int i = 0; i < maxPrint; ++i) {
  //     appendChar(' ');
  //     float v = (dtype == 0) ? clustererNN.mInputData_16[baseIdx + i].ToFloat()
  //                            : clustererNN.mInputData_32[baseIdx + i];
  //     appendFloat(v);
  //     if (pos > (MAX_CHARS - 32)) { appendStr(" ..."); break; }
  //   }
  //
  //   buffer[pos] = 0;
  //   printf("%s\n", buffer);
  // }

  tpc::ClusterNative myCluster;
  bool rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
  if (clustererNN.mNnClusterizerUseClassification) {
    rejectCluster |= (clustererNN.mOutputDataClass[peakIndex] <= 0);
  }
  if (rejectCluster) {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }

  uint32_t rowIndex = 0;
  if (clusterOut != nullptr) {
    rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
      clusterer,
      myCluster,
      peak.row(),
      clusterer.mNMaxClusterPerRow,
      clusterer.mPclusterInRow,
      clusterOut);
    if (clusterer.mPclusterPosInRow != nullptr) {
      clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
    }
  } else if (clusterer.mPclusterPosInRow) {
    rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
  }
  CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow));
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishClass2Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  if (glo_idx >= (uint32_t)clustererNN.mNnClusterizerBatchedMode) {
    return;
  }

  uint32_t maxClusterNum = clusterer.mPmemory->counters.nClusters;
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfChargePos peak = clusterer.mPfilteredPeakPositions[CAMath::Min(glo_idx + batchStart, (uint32_t)(clusterer.mPmemory->counters.nClusters - 1))];
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem);
  tpc::ClusterNative* clusterOut = clusterer.mPclusterByRow;
  uint32_t full_glo_idx = glo_idx + batchStart;

  if (full_glo_idx >= maxClusterNum) {
    if (withMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect(peak, central_charge));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        peak,
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }
    return;
  }

  uint32_t model_output_index = glo_idx * clustererNN.mNnClusterizerModelReg2NumOutputNodes;

  ClusterAccumulator pc;

  if (withMC) {
    ClusterAccumulator dummy_pc;
    CPU_ONLY(labelAcc->collect(peak, central_charge));
    GPUTPCCFClusterizer::buildCluster(
      clusterer.Param().rec,
      chargeMap,
      peak,
      smem.posBcast,
      smem.buf,
      smem.innerAboveThreshold,
      &dummy_pc,
      labelAcc);
  }
  if ((clusterer.mPmemory->fragment).isOverlap(peak.time())) {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }

  // Cluster 1
  float publishPadPosition = 0.f, publishTimePosition = 0.f;
  if (dtype == 0) {
    publishPadPosition = static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_16[model_output_index].ToFloat();
    publishTimePosition = static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_16[model_output_index + 1].ToFloat();
    isBoundaryPublish(full_glo_idx, static_cast<int32_t>(peak.row()), publishPadPosition, publishTimePosition);
    pc.setFull(central_charge * clustererNN.mOutputDataReg2_16[model_output_index + 8].ToFloat(),
               publishPadPosition,
               clustererNN.mOutputDataReg2_16[model_output_index + 4].ToFloat(),
               (clusterer.mPmemory->fragment).start + publishTimePosition,
               clustererNN.mOutputDataReg2_16[model_output_index + 6].ToFloat(),
               clustererNN.mClusterFlags[2 * glo_idx],
               clustererNN.mClusterFlags[2 * glo_idx + 1]);
  } else if (dtype == 1) {
    publishPadPosition = static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_32[model_output_index];
    publishTimePosition = static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_32[model_output_index + 1];
    isBoundaryPublish(full_glo_idx, static_cast<int32_t>(peak.row()), publishPadPosition, publishTimePosition);
    pc.setFull(central_charge * clustererNN.mOutputDataReg2_32[model_output_index + 8],
               publishPadPosition,
               clustererNN.mOutputDataReg2_32[model_output_index + 4],
               (clusterer.mPmemory->fragment).start + publishTimePosition,
               clustererNN.mOutputDataReg2_32[model_output_index + 6],
               clustererNN.mClusterFlags[2 * glo_idx],
               clustererNN.mClusterFlags[2 * glo_idx + 1]);
  }

  tpc::ClusterNative myCluster;
  bool rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
  if (clustererNN.mNnClusterizerUseClassification) {
    rejectCluster |= (clustererNN.mOutputDataClass[CAMath::Min(full_glo_idx, (uint32_t)clusterer.mPmemory->counters.nClusters - 1)] <= 0);
  }
  if (rejectCluster) {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }

  uint32_t rowIndex = 0;
  if (clusterOut != nullptr) {
    rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
      clusterer,
      myCluster,
      peak.row(),
      clusterer.mNMaxClusterPerRow,
      clusterer.mPclusterInRow,
      clusterOut);
    if (clusterer.mPclusterPosInRow != nullptr) {
      clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
    }
  } else if (clusterer.mPclusterPosInRow) {
    rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
  }
  CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow));

  // Cluster 2
  if (dtype == 0) {
    publishPadPosition = static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_16[model_output_index + 1].ToFloat();
    publishTimePosition = static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_16[model_output_index + 3].ToFloat();
    isBoundaryPublish(full_glo_idx, static_cast<int32_t>(peak.row()), publishPadPosition, publishTimePosition);
    pc.setFull(central_charge * clustererNN.mOutputDataReg2_16[model_output_index + 9].ToFloat(),
               publishPadPosition,
               clustererNN.mOutputDataReg2_16[model_output_index + 5].ToFloat(),
               (clusterer.mPmemory->fragment).start + publishTimePosition,
               clustererNN.mOutputDataReg2_16[model_output_index + 7].ToFloat(),
               clustererNN.mClusterFlags[2 * glo_idx],
               clustererNN.mClusterFlags[2 * glo_idx + 1]);
  } else if (dtype == 1) {
    publishPadPosition = static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_32[model_output_index + 1];
    publishTimePosition = static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_32[model_output_index + 3];
    isBoundaryPublish(full_glo_idx, static_cast<int32_t>(peak.row()), publishPadPosition, publishTimePosition);
    pc.setFull(central_charge * clustererNN.mOutputDataReg2_32[model_output_index + 9],
               publishPadPosition,
               clustererNN.mOutputDataReg2_32[model_output_index + 5],
               (clusterer.mPmemory->fragment).start + publishTimePosition,
               clustererNN.mOutputDataReg2_32[model_output_index + 7],
               clustererNN.mClusterFlags[2 * glo_idx],
               clustererNN.mClusterFlags[2 * glo_idx + 1]);
  }

  rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
  if (clustererNN.mNnClusterizerUseClassification) {
    rejectCluster |= (clustererNN.mOutputDataClass[CAMath::Min(full_glo_idx, (uint32_t)clusterer.mPmemory->counters.nClusters - 1)] <= 0);
  }
  if (rejectCluster) {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }

  if (clusterOut != nullptr) {
    rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
      clusterer,
      myCluster,
      peak.row(),
      clusterer.mNMaxClusterPerRow,
      clusterer.mPclusterInRow,
      clusterOut);
    if (clusterer.mPclusterPosInRow != nullptr) {
      clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
    }
  } else if (clusterer.mPclusterPosInRow) {
    rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
  }
  // CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow)); // -> Is this needed? How to handle MC labels for split clusters?
}

// ---------------------------------
template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishDeconvolutionFlags>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint batchStart)
{
  // Implements identical publishing logic as the heuristic clusterizer and deconvolution kernel
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  if (glo_idx + batchStart >= clusterer.mPmemory->counters.nClusters || glo_idx >= (uint32_t)clustererNN.mNnClusterizerBatchedMode) {
    return;
  }
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfChargePos peak = clusterer.mPfilteredPeakPositions[glo_idx + batchStart];

  clustererNN.mClusterFlags[2 * glo_idx] = 0;
  clustererNN.mClusterFlags[2 * glo_idx + 1] = 0;
  for (int i = 0; i < 8; i++) {
    Delta2 d = cfconsts::InnerNeighbors[i];
    CfChargePos tmp_pos = peak.delta(d);
    PackedCharge charge = chargeMap[tmp_pos];
    clustererNN.mClusterFlags[2 * glo_idx] += (d.y != 0 && charge.isSplit());
    clustererNN.mClusterFlags[2 * glo_idx + 1] += (d.x != 0 && charge.isSplit());
  }
  for (int i = 0; i < 16; i++) {
    Delta2 d = cfconsts::OuterNeighbors[i];
    CfChargePos tmp_pos = peak.delta(d);
    PackedCharge charge = chargeMap[tmp_pos];
    clustererNN.mClusterFlags[2 * glo_idx] += (d.y != 0 && charge.isSplit() && !charge.has3x3Peak());
    clustererNN.mClusterFlags[2 * glo_idx + 1] += (d.x != 0 && charge.isSplit() && !charge.has3x3Peak());
  }
}

// THe following arithmetic is done because the network is trained with a split between IROC and OROC boundary
GPUd() int32_t GPUTPCNNClusterizerKernels::padOffset(int32_t row_ref, int32_t row_current)
{
  if (row_current < 0 || row_current >= o2::tpc::constants::MAXGLOBALPADROW) {
    return 0; // Short-circuit for out-of-bound rows
  } else {
    return (int)((GPUTPCGeometry::NPads(row_current) - GPUTPCGeometry::NPads(row_ref)) / 2);
  }
}

GPUd() int32_t GPUTPCNNClusterizerKernels::rowOffset(int32_t row, int32_t offset)
{
  return (row > 62 ? offset : 0);
}

GPUd() bool GPUTPCNNClusterizerKernels::isBoundary(int32_t row, int32_t pad, int32_t offset)
{
  if (pad < 0 || row < 0) { // Faster short-circuit
    return true;
  } else if (row < 63) {
    return (pad >= static_cast<int>(GPUTPCGeometry::NPads(row)));
  } else if (row < (63 + offset)) { // to account for the gap between IROC and OROC. Charge will be set to the boundary fill value in order to signal boundaries to the neural network
    return true;
  } else if (row < (o2::tpc::constants::MAXGLOBALPADROW + offset)) {
    return (pad >= static_cast<int>(GPUTPCGeometry::NPads(row - offset)));
  } else {
    return true;
  }
}

GPUd() bool GPUTPCNNClusterizerKernels::isBoundaryPublish(int32_t idx, int32_t row, float& pad, float& time)
{
  if (pad < 0) {
    // printf("(NN CLUS) WARNING: Boundary detected, idx = %d, pad < 0: row %d, pad %f (%d, %d), time %f (%d, %d)\n", idx, row, pad, 0, static_cast<int>(GPUTPCGeometry::NPads(row)), time, 0, TPC_MAX_FRAGMENT_LEN_GPU);
    pad = 0.f;
    return true;
  } else if (pad >= static_cast<int>(GPUTPCGeometry::NPads(row))) {
    // printf("(NN CLUS) WARNING: Boundary detected, idx = %d, pad >= static_cast<int>(GPUTPCGeometry::NPads(row): row %d, pad %f (%d, %d), time %f (%d, %d)\n", idx, row, pad, 0, static_cast<int>(GPUTPCGeometry::NPads(row)), time, 0, TPC_MAX_FRAGMENT_LEN_GPU);
    pad = static_cast<float>(GPUTPCGeometry::NPads(row) - 1);
    return true;
  } else if (time < 0) {
    // printf("(NN CLUS) WARNING: Boundary detected, idx = %d, time < 0: row %d, pad %f (%d, %d), time %f (%d, %d)\n", idx, row, pad, 0, static_cast<int>(GPUTPCGeometry::NPads(row)), time, 0, TPC_MAX_FRAGMENT_LEN_GPU);
    time = 0.f;
    return true;
  } else if (time >= TPC_MAX_FRAGMENT_LEN_GPU) {
    // printf("(NN CLUS) WARNING: Boundary detected, idx = %d, time >= TPC_MAX_FRAGMENT_LEN_GPU: row %d, pad %f (%d, %d), time %f (%d, %d)\n", idx, row, pad, 0, static_cast<int>(GPUTPCGeometry::NPads(row)), time, 0, TPC_MAX_FRAGMENT_LEN_GPU);
    time = static_cast<float>(TPC_MAX_FRAGMENT_LEN_GPU - 1);
    return true;
  } else {
    return false;
  }
}
