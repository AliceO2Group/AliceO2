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

/// \file DebugStreamer.h
/// \brief Definition of class for writing debug informations
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_DEBUGSTREAMER_H_
#define ALICEO2_TPC_DEBUGSTREAMER_H_

#include "GPUCommonDef.h"
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
#include "CommonUtils/ConfigurableParamHelper.h"
#if defined(DEBUG_STREAMER)
#include "CommonUtils/TreeStreamRedirector.h"
#include <tbb/concurrent_unordered_map.h>
#endif
#endif

namespace o2::utils
{

/// struct defining the flags which can be used to check if a certain debug streamer is used
enum StreamFlags {
  streamdEdx = 1 << 0,          ///< stream corrections and cluster properties used for the dE/dx
  streamDigitFolding = 1 << 1,  ///< stream ion tail and saturatio information
  streamDigits = 1 << 2,        ///< stream digit information
  streamFastTransform = 1 << 3, ///< stream tpc fast transform
  streamITCorr = 1 << 4,        ///< stream ion tail correction information
  streamDistortionsSC = 1 << 5, ///< stream distortions applied in the TPC space-charge class (used for example in the tpc digitizer)
};

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

inline StreamFlags operator|(StreamFlags a, StreamFlags b)
{
  return static_cast<StreamFlags>(static_cast<int>(a) | static_cast<int>(b));
}
inline StreamFlags operator&(StreamFlags a, StreamFlags b) { return static_cast<StreamFlags>(static_cast<int>(a) & static_cast<int>(b)); }
inline StreamFlags operator~(StreamFlags a) { return static_cast<StreamFlags>(~static_cast<int>(a)); }

/// struct for setting and storing the streamer level
struct ParameterDebugStreamer : public o2::conf::ConfigurableParamHelper<ParameterDebugStreamer> {
  StreamFlags StreamLevel{}; /// flag to store what will be streamed
  O2ParamDef(ParameterDebugStreamer, "DebugStreamerParam");
};

#endif

/// class to enable streaming debug information to root files
class DebugStreamer
{

// CPU implementation of the class
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && defined(DEBUG_STREAMER)
 public:
  /// default constructor
  DebugStreamer();

  /// set the streamer i.e. create the output file and set up the streamer
  /// \param outFile output file name without .root suffix
  /// \param option RECREATE or UPDATE
  /// \param id unique id for given streamer (i.e. for defining unique streamers for each thread)
  void setStreamer(const char* outFile, const char* option, const size_t id = getCPUID());

  /// \return returns if the streamer is set
  /// \param id unique id of streamer
  bool isStreamerSet(const size_t id = getCPUID()) const { return getStreamerPtr(id); }

  /// flush TTree for given ID to disc
  /// \param id unique id of streamer
  void flush(const size_t id);

  /// flush all TTrees to disc
  void flush();

  /// \return returns streamer object for given id
  /// \param id unique id of streamer
  o2::utils::TreeStreamRedirector& getStreamer(const size_t id = getCPUID()) { return *(mTreeStreamer[id]); }

  /// \return returns streamer object
  /// \param id unique id of streamer
  o2::utils::TreeStreamRedirector* getStreamerPtr(const size_t id = getCPUID()) const;

  /// \return returns streamer level i.e. what will be written to file
  static StreamFlags getStreamFlags() { return ParameterDebugStreamer::Instance().StreamLevel; }

  ///< return returns unique ID for each CPU thread to give each thread an own output file
  static size_t getCPUID();

  /// \return returns number of trees in the streamer
  /// \param id unique id of streamer
  int getNTrees(const size_t id = getCPUID()) const;

  /// \return returns an unique branch name which is not already written in the file
  /// \param tree name of the tree for which to get a unique tree name
  /// \param id unique id of streamer
  std::string getUniqueTreeName(const char* tree, const size_t id = getCPUID()) const;

  /// set directly the debug level
  static void setStreamFlags(const StreamFlags streamFlags) { o2::conf::ConfigurableParam::setValue("DebugStreamerParam", "StreamLevel", static_cast<int>(streamFlags)); }

  /// enable specific streamer flag
  static void enableStream(const StreamFlags streamFlag);

  /// disable a specific streamer flag
  static void disableStream(const StreamFlags streamFlag);

  /// check if streamer for specific flag is enabled
  static bool checkStream(const StreamFlags streamFlag) { return ((getStreamFlags() & streamFlag) == streamFlag); }

  /// merge trees with the same content structure, but different naming
  /// \param inpFile input file containing several trees with the same content
  /// \param outFile contains the merged tree from the input file in one branch
  /// \param option setting which is used for the merging
  static void mergeTrees(const char* inpFile, const char* outFile, const char* option = "fast");

 private:
  using StreamersPerFlag = tbb::concurrent_unordered_map<size_t, std::unique_ptr<o2::utils::TreeStreamRedirector>>;
  StreamersPerFlag mTreeStreamer; ///< streamer which is used for the debugging

#else

  // empty implementation of the class for GPU or when the debug streamer is not build for CPU
 public:
  /// empty for GPU
  template <typename... Args>
  GPUd() void setStreamer(Args... args){};

  /// always false for GPU
  GPUd() static bool checkStream(const StreamFlags) { return false; }

  class StreamerDummy
  {
   public:
    GPUd() int data() const { return 0; };

    template <typename Type>
    GPUd() StreamerDummy& operator<<(Type)
    {
      return *this;
    }
  };

  GPUd() StreamerDummy getStreamer(const int id = 0) const { return StreamerDummy{}; };

  template <typename Type>
  GPUd() StreamerDummy getUniqueTreeName(Type, const int id = 0) const
  {
    return StreamerDummy{};
  }

  GPUd() void flush() const {};

#endif
};

} // namespace o2::utils

#endif
