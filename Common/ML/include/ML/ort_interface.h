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

/// \file     ort_interface.h
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
/// \brief    A header library for loading ONNX models and inferencing them on CPU and GPU

#ifndef O2_ML_ONNX_INTERFACE_H
#define O2_ML_ONNX_INTERFACE_H

// C++ and system includes
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>

// O2 includes
#include "GPUORTFloat16.h"
#include "Framework/Logger.h"

namespace o2
{

namespace ml
{

class OrtModel
{

  public:
    // Constructor
    OrtModel() = default;
    OrtModel(std::unordered_map<std::string, std::string> optionsMap){ reset(optionsMap); }
    void init(std::unordered_map<std::string, std::string> optionsMap){ reset(optionsMap); }
    void reset(std::unordered_map<std::string, std::string>);

    virtual ~OrtModel() = default;

    // Conversion
    template<class I, class O>
    std::vector<O> v2v(std::vector<I>&, bool = true);

    // Inferencing
    template<class I, class O> // class I is the input data type, e.g. float, class O is the output data type, e.g. OrtDataType::Float16_t from O2/Common/ML/include/ML/GPUORTFloat16.h
    std::vector<O> inference(std::vector<I>&);

    template<class I, class O> // class I is the input data type, e.g. float, class O is the output data type, e.g. O2::gpu::OrtDataType::Float16_t from O2/GPU/GPUTracking/ML/convert_float16.h
    std::vector<O> inference(std::vector<std::vector<I>>&);

    // template<class I, class T, class O> // class I is the input data type, e.g. float, class T the throughput data type and class O is the output data type
    // std::vector<O> inference(std::vector<I>&);

    // Reset session
    void resetSession();

    std::vector<std::vector<int64_t>> getNumInputNodes() const { return mInputShapes; }
    std::vector<std::vector<int64_t>> getNumOutputNodes() const { return mOutputShapes; }
    std::vector<std::string> getInputNames() const { return mInputNames; }
    std::vector<std::string> getOutputNames() const { return mOutputNames; }

    void setActiveThreads(int threads) { intraOpNumThreads = threads; }

  private:

    // ORT variables -> need to be hidden as Pimpl
    struct OrtVariables;
    OrtVariables* pImplOrt;

    // Input & Output specifications of the loaded network
    std::vector<const char*> inputNamesChar, outputNamesChar;
    std::vector<std::string> mInputNames, mOutputNames;
    std::vector<std::vector<int64_t>> mInputShapes, mOutputShapes;

    // Environment settings
    std::string modelPath, device = "cpu", dtype = "float"; // device options should be cpu, rocm, migraphx, cuda
    int intraOpNumThreads = 0, deviceId = 0, enableProfiling = 0, loggingLevel = 0, allocateDeviceMemory = 0, enableOptimizations = 0;

    std::string printShape(const std::vector<int64_t>&);

};

} // namespace ml

} // namespace ml

#endif // O2_ML_ORT_INTERFACE_H