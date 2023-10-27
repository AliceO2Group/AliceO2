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
  streamdEdx = 1 << 0,                  ///< stream corrections and cluster properties used for the dE/dx
  streamDigitFolding = 1 << 1,          ///< stream ion tail and saturatio information
  streamDigits = 1 << 2,                ///< stream digit information
  streamFastTransform = 1 << 3,         ///< stream tpc fast transform
  streamITCorr = 1 << 4,                ///< stream ion tail correction information
  streamDistortionsSC = 1 << 5,         ///< stream distortions applied in the TPC space-charge class (used for example in the tpc digitizer)
  streamUpdateTrack = 1 << 6,           ///< stream update track informations
  streamRejectCluster = 1 << 7,         ///< stream cluster rejection informations
  streamMergeBorderTracksBest = 1 << 8, ///< stream MergeBorderTracks best track
  streamTimeSeries = 1 << 9,            ///< stream tpc DCA debug tree
  streamMergeBorderTracksAll = 1 << 10, ///< stream MergeBorderTracks all tracks
  streamFlagsCount = 11                 ///< total number of streamers
};

enum SamplingTypes {
  sampleAll = 0,      ///< use all data (default)
  sampleRandom = 1,   ///< sample randomly every n points
  sampleID = 2,       ///< sample every n IDs (per example track)
  sampleIDGlobal = 3, ///< in case different streamers have access to the same IDs use this gloabl ID
  sampleWeights = 4,  ///< perform sampling on weights, defined where the streamer is called
  sampleTsallis = 5,  ///< perform sampling on tsallis pdf
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
  int streamLevel{};                                                                              /// flag to store what will be streamed
  SamplingTypes samplingType[StreamFlags::streamFlagsCount]{};                                    ///< sampling type for each streamer (default = SamplingTypes::sampleAll)
  float samplingFrequency[StreamFlags::streamFlagsCount]{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}; ///< frequency which is used for the sampling (0.1 -> 10% is written if sampling is used)
  int sampleIDGlobal[StreamFlags::streamFlagsCount]{};                                            ///< storage of reference streamer used for sampleIDFromOtherStreamer
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

  static DebugStreamer* instance()
  {
    static DebugStreamer streamer;
    return &streamer;
  }

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

  /// \return returns streamer object for given id
  /// \param outFile output file name without .root suffix
  /// \param option RECREATE or UPDATE
  /// \param id unique id of streamer
  o2::utils::TreeStreamRedirector& getStreamer(const char* outFile, const char* option, const size_t id = getCPUID());

  /// \return returns streamer object
  /// \param id unique id of streamer
  o2::utils::TreeStreamRedirector* getStreamerPtr(const size_t id = getCPUID()) const;

  /// \return returns streamer level i.e. what will be written to file
  static StreamFlags getStreamFlags() { return static_cast<StreamFlags>(ParameterDebugStreamer::Instance().streamLevel); }

  /// \return returns sampling type and sampling frequency for given streamer
  static std::pair<SamplingTypes, float> getSamplingTypeFrequency(const StreamFlags streamFlag);

  /// \return returns sampling type and sampling frequency for given streamer
  static float getSamplingFrequency(const StreamFlags streamFlag) { return getSamplingTypeFrequency(streamFlag).second; }

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
  static void setStreamFlags(const StreamFlags streamFlags) { o2::conf::ConfigurableParam::setValue("DebugStreamerParam", "streamLevel", static_cast<int>(streamFlags)); }

  /// enable specific streamer flag
  static void enableStream(const StreamFlags streamFlag);

  /// disable a specific streamer flag
  static void disableStream(const StreamFlags streamFlag);

  /// check if streamer for specific flag is enabled
  /// \param samplingID optional index of the data which is streamed in to perform sampling on this index
  /// \param weight weight which can be used to perform some weightes sampling
  static bool checkStream(const StreamFlags streamFlag, const size_t samplingID = -1, const float weight = 1);

  /// merge trees with the same content structure, but different naming
  /// \param inpFile input file containing several trees with the same content
  /// \param outFile contains the merged tree from the input file in one branch
  /// \param option setting which is used for the merging
  static void mergeTrees(const char* inpFile, const char* outFile, const char* option = "fast");

  /// \return returns integer index for given streamer flag
  static int getIndex(const StreamFlags streamFlag);

  /// get random value between min and max
  static float getRandom(float min = 0, float max = 1);

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
  GPUd() static bool checkStream(const StreamFlags, const int samplingID = 0) { return false; }

  GPUd() static DebugStreamer* instance() { return nullptr; }

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

  /// empty for GPU
  template <typename... Args>
  GPUd() StreamerDummy getStreamer(Args... args) const
  {
    return StreamerDummy{};
  };

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
