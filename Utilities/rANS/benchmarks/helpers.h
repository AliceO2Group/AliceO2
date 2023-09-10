// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   helpers.h
/// @author Michael Lettrich
/// @brief  common functionality for rANS benchmarks.

#ifndef RANS_BENCHMARKS_HELPERS_H_
#define RANS_BENCHMARKS_HELPERS_H_

#include "rANS/internal/common/defines.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

#ifdef RANS_ENABLE_JSON
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#endif // RANS_ENABLE_JSON
#include <fairlogger/Logger.h>
#include <algorithm>

#include "rANS/internal/common/exceptions.h"

#ifdef RANS_PARALLEL_STL
#include <execution>
#endif

#ifdef RANS_ENABLE_JSON
struct TPCCompressedClusters {

  TPCCompressedClusters() = default;
  TPCCompressedClusters(size_t nTracks,
                        size_t nAttachedClusters,
                        size_t nUnattachedClusters,
                        size_t nAttachedClustersReduced) : nTracks(nTracks),
                                                           nAttachedClusters(nAttachedClusters),
                                                           nUnattachedClusters(nUnattachedClusters),
                                                           nAttachedClustersReduced(nAttachedClustersReduced)
  {
    qTotA.resize(this->nAttachedClusters);
    qMaxA.resize(this->nAttachedClusters);
    flagsA.resize(this->nAttachedClusters);
    rowDiffA.resize(this->nAttachedClustersReduced);
    sliceLegDiffA.resize(this->nAttachedClustersReduced);
    padResA.resize(this->nAttachedClustersReduced);
    timeResA.resize(this->nAttachedClustersReduced);
    sigmaPadA.resize(this->nAttachedClusters);
    sigmaTimeA.resize(this->nAttachedClusters);

    qPtA.resize(this->nTracks);
    rowA.resize(this->nTracks);
    sliceA.resize(this->nTracks);
    timeA.resize(this->nTracks);
    padA.resize(this->nTracks);

    qTotU.resize(this->nUnattachedClusters);
    qMaxU.resize(this->nUnattachedClusters);
    flagsU.resize(this->nUnattachedClusters);
    padDiffU.resize(this->nUnattachedClusters);
    timeDiffU.resize(this->nUnattachedClusters);
    sigmaPadU.resize(this->nUnattachedClusters);
    sigmaTimeU.resize(this->nUnattachedClusters);

    nTrackClusters.resize(this->nTracks);
    nSliceRowClusters.resize(this->nSliceRows);
  };

  size_t nTracks = 0;
  size_t nAttachedClusters = 0;
  size_t nUnattachedClusters = 0;
  size_t nAttachedClustersReduced = 0;
  size_t nSliceRows = 36 * 152;

  std::vector<uint16_t> qTotA{};
  std::vector<uint16_t> qMaxA{};
  std::vector<uint8_t> flagsA{};
  std::vector<uint8_t> rowDiffA{};
  std::vector<uint8_t> sliceLegDiffA{};
  std::vector<uint16_t> padResA{};
  std::vector<uint32_t> timeResA{};
  std::vector<uint8_t> sigmaPadA{};
  std::vector<uint8_t> sigmaTimeA{};

  std::vector<uint8_t> qPtA{};
  std::vector<uint8_t> rowA{};
  std::vector<uint8_t> sliceA{};
  std::vector<uint32_t> timeA{};
  std::vector<uint16_t> padA{};

  std::vector<uint16_t> qTotU{};
  std::vector<uint16_t> qMaxU{};
  std::vector<uint8_t> flagsU{};
  std::vector<uint16_t> padDiffU{};
  std::vector<uint32_t> timeDiffU{};
  std::vector<uint8_t> sigmaPadU{};
  std::vector<uint8_t> sigmaTimeU{};

  std::vector<uint16_t> nTrackClusters{};
  std::vector<uint32_t> nSliceRowClusters{};
};

class TPCJsonHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, TPCJsonHandler>
{
 private:
  class CurrentVector
  {
   public:
    CurrentVector() = default;

    template <typename T>
    CurrentVector(std::vector<T>& vec) : mVectorConcept{std::make_unique<VectorWrapper<T>>(vec)} {};

    inline void push_back(unsigned i) { mVectorConcept->push_back(i); };

    struct VectorConcept {
      virtual ~VectorConcept() = default;
      virtual void push_back(unsigned i) = 0;
    };

    template <typename T>
    struct VectorWrapper : VectorConcept {
      VectorWrapper(std::vector<T>& vector) : mVector(vector){};

      void push_back(unsigned i) override { mVector.push_back(static_cast<T>(i)); };

     private:
      std::vector<T>& mVector{};
    };

    std::unique_ptr<VectorConcept> mVectorConcept{};
  };

 public:
  bool Null() { return true; };
  bool Bool(bool b) { return true; };
  bool Int(int i) { return this->Uint(static_cast<unsigned>(i)); };
  bool Uint(unsigned i)
  {
    mCurrentVector.push_back(i);
    return true;
  };
  bool Int64(int64_t i) { return this->Uint(static_cast<unsigned>(i)); };
  bool Uint64(uint64_t i) { return this->Uint(static_cast<unsigned>(i)); };
  bool Double(double d) { return this->Uint(static_cast<unsigned>(d)); };
  bool RawNumber(const Ch* str, rapidjson::SizeType length, bool copy) { return true; };
  bool String(const Ch* str, rapidjson::SizeType length, bool copy) { return true; };
  bool StartObject() { return true; };
  bool Key(const Ch* str, rapidjson::SizeType length, bool copy)
  {
    if (str == std::string{"qTotA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.qTotA};
    } else if (str == std::string{"qMaxA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.qMaxA};
    } else if (str == std::string{"flagsA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.flagsA};
    } else if (str == std::string{"rowDiffA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.rowDiffA};
    } else if (str == std::string{"sliceLegDiffA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.sliceLegDiffA};
    } else if (str == std::string{"padResA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.padResA};
    } else if (str == std::string{"timeResA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.timeResA};
    } else if (str == std::string{"sigmaPadA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.sigmaPadA};
    } else if (str == std::string{"sigmaTimeA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.sigmaTimeA};
    } else if (str == std::string{"qPtA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.qPtA};
    } else if (str == std::string{"rowA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.rowA};
    } else if (str == std::string{"sliceA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.sliceA};
    } else if (str == std::string{"timeA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.timeA};
    } else if (str == std::string{"padA"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.padA};
    } else if (str == std::string{"qTotU"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.qTotU};
    } else if (str == std::string{"qMaxU"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.qMaxU};
    } else if (str == std::string{"flagsU"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.flagsU};
    } else if (str == std::string{"padDiffU"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.padDiffU};
    } else if (str == std::string{"timeDiffU"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.timeDiffU};
    } else if (str == std::string{"sigmaPadU"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.sigmaPadU};
    } else if (str == std::string{"sigmaTimeU"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.sigmaTimeU};
    } else if (str == std::string{"nTrackClusters"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.nTrackClusters};
    } else if (str == std::string{"nSliceRowClusters"}) {
      LOGP(info, "parsing {}", str);
      mCurrentVector = CurrentVector{mCompressedClusters.nSliceRowClusters};
    } else {
      throw o2::rans::IOError(fmt::format("invalid key: {}", str));
      return false;
    }
    return true;
  };
  bool EndObject(rapidjson::SizeType memberCount)
  {
    LOGP(info, "parsed {} objects", memberCount);
    return true;
  };
  bool StartArray()
  {
    return true;
  };
  bool EndArray(rapidjson::SizeType elementCount)
  {
    LOGP(info, "parsed {} elements", elementCount);
    return true;
  };

  TPCCompressedClusters release() && { return std::move(mCompressedClusters); };

 private:
  TPCCompressedClusters mCompressedClusters;
  CurrentVector mCurrentVector;
};

TPCCompressedClusters readFile(const std::string& filename)
{
  TPCCompressedClusters compressedClusters;
  std::ifstream is(filename, std::ios_base::in);
  if (is) {
    rapidjson::IStreamWrapper isWrapper{is};
    // rapidjson::Document document;
    TPCJsonHandler handler;
    rapidjson::Reader reader;
    reader.Parse(isWrapper, handler);

    compressedClusters = std::move(handler).release();

    compressedClusters.nAttachedClusters = compressedClusters.qTotA.size();
    compressedClusters.nAttachedClustersReduced = compressedClusters.rowDiffA.size();
    compressedClusters.nUnattachedClusters = compressedClusters.qTotA.size();
    compressedClusters.nTracks = compressedClusters.nTrackClusters.size();
    is.close();
  } else {
    throw o2::rans::IOError(fmt::format("Could not open file {}", filename));
  }
  return compressedClusters;
};
#endif // RANS_ENABLE_JSON

template <typename source_T, typename stream_T = uint32_t>
struct EncodeBuffer {

  EncodeBuffer() = default;
  EncodeBuffer(size_t sourceSize) : buffer(2 * sourceSize), literals()
  {
    literals.reserve(sourceSize + 32);
    literalsEnd = literals.data();
    encodeBufferEnd = buffer.data();
  };

  std::vector<stream_T> buffer{};
  std::vector<source_T> literals{};
  stream_T* encodeBufferEnd{buffer.data()};
  source_T* literalsEnd{literals.data()};
};

template <typename source_T>
struct DecodeBuffer {

  DecodeBuffer() = default;
  DecodeBuffer(size_t sourceSize) : buffer(sourceSize){};

  template <typename T>
  bool operator==(const T& correct)
  {
#ifdef RANS_PARALLEL_STL
    return std::equal(std::execution::par_unseq, buffer.begin(), buffer.end(), std::begin(correct), std::end(correct));
#else
    return std::equal(buffer.begin(), buffer.end(), std::begin(correct), std::end(correct));
#endif // RANS_PARALLEL_STL
  }

  std::vector<source_T> buffer{};
};

#endif /* RANS_BENCHMARKS_HELPERS_H_ */