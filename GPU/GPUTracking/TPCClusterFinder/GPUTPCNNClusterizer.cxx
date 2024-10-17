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

  if(clusterer.OrtOptions["dtype"].find("32") != std::string::npos){
    GPUTPCNNClusterizer::nn_clusterizer<float>(nBlocks, nThreads, iBlock, iThread, clusterer, clusterer.mPmemory->fragment, smem, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
  } else if(clusterer.OrtOptions["dtype"].find("16") != std::string::npos) {
    GPUTPCNNClusterizer::nn_clusterizer<OrtDataType::Float16_t>(nBlocks, nThreads, iBlock, iThread, clusterer, clusterer.mPmemory->fragment, smem, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
  } else {
    LOG(fatal) << "Unsupported data type for neural network clusterizer!";
  }
  // tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;
// 
  // GPUTPCNNClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, smem, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
}

int GPUTPCNNClusterizer::padOffset(int row_ref, int row_current, const GPUTPCGeometry& geo)
{
  return (int)((geo.NPads(row_current) - geo.NPads(row_ref)) / 2);
}

int GPUTPCNNClusterizer::rowOffset(int row, int global_shift)
{
  return (row > 62 ? global_shift : 0);
}

// ---------------------------------
bool GPUTPCNNClusterizer::isBoundary(int row, int pad, int global_shift, const GPUTPCGeometry& geo)
{
  if (pad < 0 || row < 0) { // Faster short-circuit
    return true;
  } else if (row <= 62) {
    // if (pad < (geo.NPads(o2):tpc::constants::MAXGLOBALPADROW-1] - geo.NPads(row)) / 2 || pad > (geo.NPads(o2):tpc::constants::MAXGLOBALPADROW-1] + geo.NPads(row)) / 2) {
    //   return true;
    // } else {
    //   return false;
    // }
    if (pad < 0 || pad > geo.NPads(row)) {
      return true;
    } else {
      return false;
    }
  } else if (row <= 62 + global_shift) { // to account for the gap between IROC and OROC. Charge will be set to -1 in order to signal boundary to the neural network
    return true;
  } else if (row <= o2::tpc::constants::MAXGLOBALPADROW-1 + global_shift) {
    //if (pad < (geo.NPads(o2):tpc::constants::MAXGLOBALPADROW-1] - geo.NPads(row)- global_shift]) / 2 || pad > (geo.NPads(o2):tpc::constants::MAXGLOBALPADROW-1] + geo.NPads(row)- global_shift]) / 2) {
    //  return true;
    //} else {
    //  return false;
    //}
    if (pad < 0 || pad > geo.NPads(row)) {
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}

template <class T>
GPUd() void GPUTPCNNClusterizer::nn_clusterizer(int nBlocks, int nThreads, int iBlock, int iThread,
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
                                          uint* clusterPosInRow){

    uint glo_idx = get_global_id(0) * clusterer.nnClusterizerBatchedMode;
    if(glo_idx >= clusternum){
      return;
    }

    std::vector<float> central_charges(clusterer.nnClusterizerBatchedMode, -1.f);
    std::vector<T> input_data(clusterer.nnClusterizerElementSize * clusterer.nnClusterizerBatchedMode, (T)-1.f);
    std::vector<ChargePos> peak_positions(clusterer.nnClusterizerBatchedMode);
    unsigned int write_idx = 0;

    for(int batch_counter = 0; batch_counter < clusterer.nnClusterizerBatchedMode; batch_counter++){

      uint cls = CAMath::Min(glo_idx + batch_counter, clusternum - 1);

      ChargePos peak = clusterer.mPfilteredPeakPositions[cls];
      int row = peak.row(), pad = peak.pad(), time = peak.time();
      float central_charge = chargeMap[peak].unpack();

      peak_positions[batch_counter] = peak;
      central_charges[batch_counter] = central_charge;

      // unsigned int batch_offset = batch_counter * clusterer.nnClusterizerElementSize;
      for(int r = -clusterer.nnClusterizerSizeInputRow; r <= clusterer.nnClusterizerSizeInputRow; r++){
        bool push_mc_label = (r == 0);
        int pad_offset = GPUTPCNNClusterizer::padOffset(row, row + r, clusterer.Param().tpcGeometry);
        int row_offset = GPUTPCNNClusterizer::rowOffset(row, clusterer.nnClusterizerSizeInputRow);
        for(int p = -clusterer.nnClusterizerSizeInputPad; p <= clusterer.nnClusterizerSizeInputPad; p++){
          push_mc_label &= (std::abs(p) < 2); // Use inner 5x5 window
          bool is_boundary = GPUTPCNNClusterizer::isBoundary(row + r + row_offset, pad + p + pad_offset, clusterer.nnClusterizerSizeInputRow, clusterer.Param().tpcGeometry);
          for(int t = -clusterer.nnClusterizerSizeInputTime; t <= clusterer.nnClusterizerSizeInputTime; t++){
            push_mc_label &= (std::abs(t) < 2); // Use inner 5x5 window
            if(!is_boundary){
              ChargePos tmp_pos(row + r, pad + p + pad_offset, time + t);
              input_data[write_idx] = (T)(chargeMap[tmp_pos].unpack() / central_charge);
              if(push_mc_label){
                ChargePos tmp_pos_mc(row, pad + p, time + t);
                CPU_ONLY(labelAcc->collect(tmp_pos, chargeMap[tmp_pos_mc].unpack()));
              }
            }
            write_idx++;
          }
        }
      }
      if(clusterer.nnClusterizerAddIndexData){
        input_data[write_idx] = (T)(clusterer.mISlice / 36.f);
        input_data[write_idx + 1] = (T)(row / 152.f);
        input_data[write_idx + 2] = (T)((float)pad / clusterer.Param().tpcGeometry.NPads(row));
        write_idx+=3;
        // if(idx == 100){
        //   LOG(info) << "[" << input_data[input_data.size()-3] << ", " << input_data[input_data.size()-2] << ", " << input_data[input_data.size()-1] << "]";
        // }
      }
    }

    std::vector<int> index_class_2;
    std::vector<float> out_class = clusterer.model_class.inference<T,float>(input_data);
    // LOG(info) << "input_data.size(): " << input_data.size() << "; write_idx: " << write_idx << "; out_class.size(): " << out_class.size();
    int num_output_classes = clusterer.model_class.getNumOutputNodes()[0][1];

    if(num_output_classes > 1){
      std::vector<float> tmp_out_class(clusterer.nnClusterizerBatchedMode);
      for(int cls_idx = 0; cls_idx < clusterer.nnClusterizerBatchedMode; cls_idx++){
        auto elem_iterator = out_class.begin() + (unsigned int)(cls_idx*num_output_classes);
        tmp_out_class[cls_idx] = std::distance(elem_iterator, std::max_element(elem_iterator, elem_iterator+num_output_classes)) - 1; // -1 since 2-class classifier will have 3 outputs: classes 0, 1, 2
        if(tmp_out_class[cls_idx] > 1){
          index_class_2.push_back(cls_idx);
        }
      }
      out_class = tmp_out_class;
    }

    if(!clusterer.nnClusterizerUseCFregression) {

      std::vector<float> out_reg = clusterer.model_reg_1.inference<T,float>(input_data), tmp_out_reg_2;
      if(index_class_2.size() > 0){
        std::vector<T> tmp_in_reg_2(index_class_2.size() * clusterer.nnClusterizerElementSize);
        int fill_counter = 0;
        for(int cls_idx : index_class_2){
          int from_idx = cls_idx*clusterer.nnClusterizerElementSize, to_idx = fill_counter * clusterer.nnClusterizerElementSize;
          for(int reg_idx = 0; reg_idx < clusterer.nnClusterizerElementSize; reg_idx++){
            tmp_in_reg_2[to_idx + reg_idx] = input_data[from_idx + reg_idx];
          }
          fill_counter++;
        }
        tmp_out_reg_2 = clusterer.model_reg_2.inference<T,float>(input_data);
      }

      input_data.clear();

      if((clusterer.nnClusterizerVerbosity >= 4) && glo_idx == 0){
        LOG(info) << "[CF] Classification model: " << out_class[0] << " (>? " << clusterer.nnClassThreshold << ")";
        LOG(info) << "[CF] Regression model: " << out_reg[0] << "; " << out_reg[1] << "; " << out_reg[2] << "; " << out_reg[3] << "; " << out_reg[4];
      }

      int num_outputs_1 = clusterer.model_reg_1.getNumOutputNodes()[0][1], num_outputs_2 = 0, counter_class_2_idcs = 0;
      if(num_output_classes > 1){
        num_outputs_2 = clusterer.model_reg_2.getNumOutputNodes()[0][1];
      }

      for(int element = 0; element < clusterer.nnClusterizerBatchedMode; element++) {

        if (glo_idx + element >= clusternum) {
          return;
        }

        int model_output_index = element*num_outputs_1;
        if(out_class[element] > clusterer.nnClassThreshold) {
          if((num_output_classes == 1) || ((num_output_classes > 1) && (out_class[element] < 2))) {
            // CPU_ONLY(labelAcc->collect(peak_positions[element], central_charges[element]));
            ClusterAccumulator pc;

            ClusterAccumulator dummy_pc;
            CPU_ONLY(labelAcc->collect(peak_positions[element], central_charges[element]));

            // Dummy build to push MC labels
            buildCluster(
              calib,
              chargeMap,
              peak_positions[element],
              smem.posBcast,
              smem.buf,
              smem.innerAboveThreshold,
              &dummy_pc,
              labelAcc);

            if (fragment.isOverlap(peak_positions[element].time())) {
              if (clusterPosInRow) {
                clusterPosInRow[glo_idx + element] = maxClusterPerRow;
              }
              continue;
            }

            pc.setFull(central_charges[element] * out_reg[model_output_index + 4], peak_positions[element].pad() + out_reg[model_output_index + 0], out_reg[model_output_index + 2], fragment.start + peak_positions[element].time() + out_reg[model_output_index + 1], out_reg[model_output_index + 3], 0, 0);
            // LOG(info) << "Example: " << num_outputs_1 << " " << out_reg.size() << ";; " << out_reg[model_output_index + 4] << "; " << out_reg[model_output_index + 0] << "; " << out_reg[model_output_index + 2] << "; " << out_reg[model_output_index + 1] << "; " << out_reg[model_output_index + 3];

            tpc::ClusterNative myCluster;
            bool rejectCluster = !pc.toNative(peak_positions[element], central_charges[element], myCluster, clusterer.Param());
            if (rejectCluster) {
              if(clusterer.nnClusterizerVerbosity > 3){
                LOG(warning) << "[CF] Cluster rejected!";
              }
              if (clusterPosInRow) {
                clusterPosInRow[glo_idx + element] = maxClusterPerRow;
              }
              continue;
            }

            uint rowIndex = 0;
            if (clusterByRow != nullptr) {
              rowIndex = sortIntoBuckets(
                clusterer,
                myCluster,
                peak_positions[element].row(),
                maxClusterPerRow,
                clusterInRow,
                clusterByRow);
              if (clusterPosInRow != nullptr) {
                clusterPosInRow[glo_idx + element] = rowIndex;
              }
            } else if (clusterPosInRow) {
              rowIndex = clusterPosInRow[glo_idx + element];
            }
            CPU_ONLY(labelAcc->commit(peak_positions[element].row(), rowIndex, maxClusterPerRow));
          } else {
            model_output_index = index_class_2[counter_class_2_idcs]*num_outputs_2;
            counter_class_2_idcs++;

            // Cluster 1
            CPU_ONLY(labelAcc->collect(peak_positions[element], central_charges[element]));
            ClusterAccumulator pc;

            if (fragment.isOverlap(peak_positions[element].time())) {
              if (clusterPosInRow) {
                clusterPosInRow[glo_idx + element] = maxClusterPerRow;
              }
              continue;
            }

            pc.setFull(central_charges[element] * tmp_out_reg_2[model_output_index + 8], peak_positions[element].pad() + tmp_out_reg_2[model_output_index + 4], tmp_out_reg_2[model_output_index + 2], fragment.start + peak_positions[element].time() + tmp_out_reg_2[model_output_index + 2], tmp_out_reg_2[model_output_index + 6], 0, 0);
            // LOG(info) << "Example: " << num_outputs_2 << " " << out_reg.size() << ";; " << out_reg[model_output_index + 4] << "; " << out_reg[model_output_index + 0] << "; " << out_reg[model_output_index + 2] << "; " << out_reg[model_output_index + 1] << "; " << out_reg[model_output_index + 3];

            tpc::ClusterNative myCluster;
            bool rejectCluster = !pc.toNative(peak_positions[element], central_charges[element], myCluster, clusterer.Param());
            if (rejectCluster) {
              if(clusterer.nnClusterizerVerbosity > 3){
                LOG(warning) << "[CF] Cluster rejected!";
              }
              if (clusterPosInRow) {
                clusterPosInRow[glo_idx + element] = maxClusterPerRow;
              }
              continue;
            }

            uint rowIndex = 0;
            if (clusterByRow != nullptr) {
              rowIndex = sortIntoBuckets(
                clusterer,
                myCluster,
                peak_positions[element].row(),
                maxClusterPerRow,
                clusterInRow,
                clusterByRow);
              if (clusterPosInRow != nullptr) {
                clusterPosInRow[glo_idx + element] = rowIndex;
              }
            } else if (clusterPosInRow) {
              rowIndex = clusterPosInRow[glo_idx + element];
            }
            CPU_ONLY(labelAcc->commit(peak_positions[element].row(), rowIndex, maxClusterPerRow));

            // Cluster 2
            CPU_ONLY(labelAcc->collect(peak_positions[element], central_charges[element]));
            pc.setFull(central_charges[element] * tmp_out_reg_2[model_output_index + 9], peak_positions[element].pad() + tmp_out_reg_2[model_output_index + 1], tmp_out_reg_2[model_output_index + 5], fragment.start + peak_positions[element].time() + tmp_out_reg_2[model_output_index + 3], tmp_out_reg_2[model_output_index + 7], 0, 0);
            // LOG(info) << "Example: " << num_outputs_2 << " " << out_reg.size() << ";; " << out_reg[model_output_index + 4] << "; " << out_reg[model_output_index + 0] << "; " << out_reg[model_output_index + 2] << "; " << out_reg[model_output_index + 1] << "; " << out_reg[model_output_index + 3];
            rejectCluster = !pc.toNative(peak_positions[element], central_charges[element], myCluster, clusterer.Param());
            if (rejectCluster) {
              if(clusterer.nnClusterizerVerbosity > 3){
                LOG(warning) << "[CF] Cluster rejected!";
              }
              if (clusterPosInRow) {
                clusterPosInRow[glo_idx + element] = maxClusterPerRow;
              }
              continue;
            }

            rowIndex = 0;
            if (clusterByRow != nullptr) {
              rowIndex = sortIntoBuckets(
                clusterer,
                myCluster,
                peak_positions[element].row(),
                maxClusterPerRow,
                clusterInRow,
                clusterByRow);
              if (clusterPosInRow != nullptr) {
                clusterPosInRow[glo_idx + element] = rowIndex;
              }
            } else if (clusterPosInRow) {
              rowIndex = clusterPosInRow[glo_idx + element];
            }
            CPU_ONLY(labelAcc->commit(peak_positions[element].row(), rowIndex, maxClusterPerRow));
          }
        }
      }

    } else {

      input_data.clear();
      for(int element = 0; element < clusterer.nnClusterizerBatchedMode; element++) {
        if (glo_idx + element >= clusternum) {
          return;
        }

        if(out_class[element] > clusterer.nnClassThreshold) {

          ClusterAccumulator pc;
          CPU_ONLY(labelAcc->collect(peak_positions[element], central_charges[element]));

          buildCluster(
            calib,
            chargeMap,
            peak_positions[element],
            smem.posBcast,
            smem.buf,
            smem.innerAboveThreshold,
            &pc,
            labelAcc);

          if (fragment.isOverlap(peak_positions[element].time())) {
            if (clusterPosInRow) {
              clusterPosInRow[glo_idx + element] = maxClusterPerRow;
            }
            continue;
          }
          pc.finalize(peak_positions[element], central_charges[element], fragment.start, clusterer.Param().tpcGeometry);

          tpc::ClusterNative myCluster;
          bool rejectCluster = !pc.toNative(peak_positions[element], central_charges[element], myCluster, clusterer.Param());

          if (rejectCluster) {
              if(clusterer.nnClusterizerVerbosity > 3){
                LOG(warning) << "[CF] Cluster rejected!";
              }
              if (clusterPosInRow) {
                clusterPosInRow[glo_idx + element] = maxClusterPerRow;
              }
              continue;
            }

          uint rowIndex = 0;
          if (clusterByRow != nullptr) {
            rowIndex = sortIntoBuckets(
              clusterer,
              myCluster,
              peak_positions[element].row(),
              maxClusterPerRow,
              clusterInRow,
              clusterByRow);
            if (clusterPosInRow != nullptr) {
              clusterPosInRow[glo_idx + element] = rowIndex;
            }
          } else if (clusterPosInRow) {
            rowIndex = clusterPosInRow[glo_idx + element];
          }

          CPU_ONLY(labelAcc->commit(peak_positions[element].row(), rowIndex, maxClusterPerRow));
        }
      }
    }

    if(clusterer.nnClusterizerVerbosity > 4){
      LOG(info) << "[CF] Clusterization done!";
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