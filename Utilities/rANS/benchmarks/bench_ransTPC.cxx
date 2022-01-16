#include <vector>
#include <cstring>
#include <execution>
#include <iterator>
#include <iostream>

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

#include <boost/program_options.hpp>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <fairlogger/Logger.h>

#include "rANS/rans.h"
#include "rANS/LiteralSIMDEncoder.h"
#include "rANS/LiteralSIMDDecoder.h"

namespace bpo = boost::program_options;

template <typename source_T>
double_t computeExpectedCodewordLength(const o2::rans::FrequencyTable& frequencies, const o2::rans::RenormedFrequencyTable& rescaled)
{

  using symbol_t = o2::rans::symbol_t;
  using count_t = o2::rans::count_t;
  double_t expectedCodewordLength = 0;
  count_t trueIncompressibleFrequency = frequencies.getIncompressibleSymbolFrequency();

  auto getRescaledFrequency = [&rescaled](symbol_t sourceSymbol) {
    if (sourceSymbol >= rescaled.getMinSymbol() && sourceSymbol <= rescaled.getMaxSymbol()) {
      return rescaled[sourceSymbol];
    } else {
      return static_cast<count_t>(0);
    }
  };

  // all "normal symbols"
  for (symbol_t sourceSymbol = frequencies.getMinSymbol(); sourceSymbol <= frequencies.getMaxSymbol(); ++sourceSymbol) {

    const count_t frequency = frequencies[sourceSymbol];
    if (frequency) {
      const count_t rescaledFrequency = getRescaledFrequency(sourceSymbol);

      const double_t trueProbability = static_cast<double_t>(frequency) / frequencies.getNumSamples();

      if (rescaledFrequency) {
        const double_t rescaledProbability = static_cast<double_t>(rescaledFrequency) / rescaled.getNumSamples();
        expectedCodewordLength -= trueProbability * std::log2(rescaledProbability);
      } else {
        trueIncompressibleFrequency += frequency;
      }
    }
  }
  // incompressibleSymbol:
  const double_t trueProbability = static_cast<double_t>(trueIncompressibleFrequency) / frequencies.getNumSamples();
  const double_t rescaledProbability = static_cast<double_t>(rescaled.getIncompressibleSymbolFrequency()) / rescaled.getNumSamples();

  expectedCodewordLength -= trueProbability * std::log2(rescaledProbability);
  expectedCodewordLength += trueProbability * std::log2(o2::rans::internal::toBits(sizeof(source_T)));

  return expectedCodewordLength;
};

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
  std::vector<int32_t> timeA{};
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
      virtual ~VectorConcept(){};
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
      throw std::runtime_error(fmt::format("invalid key: {}", str));
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
    throw std::runtime_error(fmt::format("Could not open file {}", filename));
  }
  return compressedClusters;
};

template <typename source_T, typename stream_T = uint32_t>
struct EncodeBuffer {

  EncodeBuffer() = default;
  EncodeBuffer(size_t sourceSize) : buffer(2 * sourceSize), literals()
  {
    literals.reserve(sourceSize + 32);
  };

  std::vector<stream_T> buffer{};
  std::vector<source_T> literals{};
  stream_T* encodeIter{buffer.data()};
};

template <typename source_T>
struct DecodeBuffer {

  DecodeBuffer() = default;
  DecodeBuffer(size_t sourceSize) : buffer(sourceSize){};

  template <typename T>
  bool operator==(const T& correct)
  {
    return std::equal(std::execution::par_unseq, buffer.begin(), buffer.end(), std::begin(correct), std::end(correct));
  }

  std::vector<source_T> buffer{};
};

template <typename encoder_T, typename decoder_T>
void ransEncodeDecode(const std::string& name, const std::vector<typename encoder_T::source_t>& inputData, rapidjson::Writer<rapidjson::OStreamWrapper>& writer)
{
  using namespace o2;
  using source_t = typename encoder_T::source_t;
  rans::internal::RANSTimer timer{};
  writer.Key(name.c_str());
  writer.StartObject();

  EncodeBuffer<source_t> encodeBuffer{inputData.size()};
  DecodeBuffer<source_t> decodeBuffer{inputData.size()};

  auto makeFrequencyTable = [](const auto& input) {
    if constexpr (sizeof(source_t) < 4) {
      // 8it and 16 bit types:
      return rans::makeFrequencyTableFromSamples(input.begin(), input.end(), std::numeric_limits<source_t>::min(), std::numeric_limits<source_t>::max());
    } else {
      return rans::makeFrequencyTableFromSamples(input.begin(), input.end());
    }
  };

  writer.Key("Timing");
  writer.StartObject();

  LOGP(info, "processing: {} (nItems: {}, size: {} MiB)", name, inputData.size(), inputData.size() * sizeof(source_t) / 1024.0 / 1024.0);
  timer.start();

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif

  auto frequencyTable = makeFrequencyTable(inputData);

#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  timer.stop();
  writer.Key("FrequencyTable");
  writer.Double(timer.getDurationMS());
  LOGP(info, "Built Frequency Table in {} ms", timer.getDurationMS());

  timer.start();
  auto renormedFrequencyTable = rans::renormCutoffIncompressible(frequencyTable);

  timer.stop();
  writer.Key("Renorming");
  writer.Double(timer.getDurationMS());
  LOGP(info, "Renormed Frequency Table in {} ms", timer.getDurationMS());
  timer.start();
  encoder_T encoder{renormedFrequencyTable};
  timer.stop();
  writer.Key("Encoder");
  writer.Double(timer.getDurationMS());
  LOGP(info, "Built Encoder in {} ms", timer.getDurationMS());
  timer.start();
  encodeBuffer.encodeIter = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.encodeIter, encodeBuffer.literals);
  timer.stop();
  writer.Key("Encoding");
  writer.Double(timer.getDurationMS());
  LOGP(info, "Encoded {} Bytes in {} ms", inputData.size() * sizeof(source_t), timer.getDurationMS());

  timer.start();
  decoder_T decoder{renormedFrequencyTable};
  timer.stop();
  writer.Key("Decoder");
  writer.Double(timer.getDurationMS());
  LOGP(info, "Built Decoder in {} ms", timer.getDurationMS());
  timer.start();
  decoder.process(encodeBuffer.encodeIter, decodeBuffer.buffer.data(), inputData.size(), encodeBuffer.literals);
  timer.stop();
  writer.Key("Decoding");
  writer.Double(timer.getDurationMS());
  LOGP(info, "Decoded {} Bytes in {} ms", inputData.size() * sizeof(source_t), timer.getDurationMS());

  if (!(decodeBuffer == inputData)) {
    LOGP(warning, "Missmatch between original and decoded Message");
  }
  LOG(info) << "finished: " << name;

  writer.EndObject(); // Timing

  //   writer.Key("Items");
  //   writer.Key("ItemSize") : writer.Key();

  //   st.SetItemsProcessed(static_cast<int64_t>(sourceMessage.size()) * static_cast<int64_t>(st.iterations()));
  //   st.SetBytesProcessed(static_cast<int64_t>(sourceMessage.size()) * sizeof(typename fixture_T::source_t) * static_cast<int64_t>(st.iterations()));

  // Frequency Table
  //##########################
  writer.Key("FrequencyTable");
  writer.StartObject();
  writer.Key("nSamples");
  writer.Uint64(frequencyTable.getNumSamples());
  writer.Key("Min");
  writer.Int(frequencyTable.getMinSymbol());
  writer.Key("Max");
  writer.Int(frequencyTable.getMaxSymbol());
  writer.Key("alphabetRangeBits");
  writer.Int(frequencyTable.getAlphabetRangeBits());
  writer.Key("nUsedAlphabetSymbols");
  writer.Uint(frequencyTable.getNUsedAlphabetSymbols());
  writer.Key("IncompressibleFrequency");
  writer.Uint(frequencyTable.getIncompressibleSymbolFrequency());
  writer.EndObject(); // FrequencyTable

  // ResacledFrequencies
  //##########################
  writer.Key("ResacledFrequencies");
  writer.StartObject();
  writer.Key("nSamples");
  writer.Uint64(renormedFrequencyTable.getNumSamples());
  writer.Key("Min");
  writer.Int(renormedFrequencyTable.getMinSymbol());
  writer.Key("Max");
  writer.Int(renormedFrequencyTable.getMaxSymbol());
  writer.Key("alphabetRangeBits");
  writer.Int(renormedFrequencyTable.getAlphabetRangeBits());
  writer.Key("nUsedAlphabetSymbols");
  writer.Uint(renormedFrequencyTable.getNUsedAlphabetSymbols());
  writer.Key("IncompressibleFrequency");
  writer.Uint(renormedFrequencyTable.getIncompressibleSymbolFrequency());
  writer.Key("RenormingBits");
  writer.Uint(renormedFrequencyTable.getRenormingBits());
  writer.EndObject(); // ResacledFrequencies

  // Message Properties
  //##########################
  writer.Key("Message");
  writer.StartObject();
  writer.Key("Size");
  writer.Uint64(inputData.size());
  writer.Key("SymbolSize");
  writer.Uint(sizeof(source_t));
  writer.Key("Entropy");
  writer.Double(o2::rans::computeEntropy(frequencyTable));
  writer.Key("ExpectedCodewordLength");
  writer.Double(computeExpectedCodewordLength<source_t>(frequencyTable, renormedFrequencyTable));
  writer.EndObject(); // Message

  // Message Properties
  //##########################
  writer.Key("Compression");
  writer.StartObject();
  writer.Key("EncodeBufferSize");
  writer.Uint64(std::distance(encodeBuffer.buffer.data(), encodeBuffer.encodeIter) * sizeof(uint32_t));
  writer.Key("LiteralSize");
  writer.Uint64(encodeBuffer.literals.size() * sizeof(source_t));
  writer.EndObject(); // Compression

  writer.EndObject(); // Encode/Decode Run
};

// template <typename encoder_T, typename decoder_T>
// void ransEncodeDecode(){};

template <template <class> class encoder_T, template <class> class decoder_T>
void encodeTPC(const std::string& name, const TPCCompressedClusters& compressedClusters, bool mergeColumns, rapidjson::Writer<rapidjson::OStreamWrapper>& writer)
{
  writer.Key(name.c_str());
  writer.StartObject();
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("qTotA", compressedClusters.qTotA, writer);
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("qMaxA", compressedClusters.qMaxA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("flagsA", compressedClusters.flagsA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("rowDiffA", compressedClusters.rowDiffA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("sliceLegDiffA", compressedClusters.sliceLegDiffA, writer);
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("padResA", compressedClusters.padResA, writer);
  // ransEncodeDecode<encoder_T<uint32_t>, decoder_T<uint32_t>>("timeResA", compressedClusters.timeResA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("sigmaPadA", compressedClusters.sigmaPadA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("sigmaTimeA", compressedClusters.sigmaTimeA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("qPtA", compressedClusters.qPtA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("rowA", compressedClusters.rowA, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("sliceA", compressedClusters.sliceA, writer);
  // ransEncodeDecode<encoder_T<int32_t>, decoder_T<int32_t>>("timeA", compressedClusters.timeA, writer);
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("padA", compressedClusters.padA, writer);
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("qTotU", compressedClusters.qTotU, writer);
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("qMaxU", compressedClusters.qMaxU, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("flagsU", compressedClusters.flagsU, writer);
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("padDiffU", compressedClusters.padDiffU, writer);
  ransEncodeDecode<encoder_T<uint32_t>, decoder_T<uint32_t>>("timeDiffU", compressedClusters.timeDiffU, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("sigmaPadU", compressedClusters.sigmaPadU, writer);
  ransEncodeDecode<encoder_T<uint8_t>, decoder_T<uint8_t>>("sigmaTimeU", compressedClusters.sigmaTimeU, writer);
  ransEncodeDecode<encoder_T<uint16_t>, decoder_T<uint16_t>>("nTrackClusters", compressedClusters.nTrackClusters, writer);
  ransEncodeDecode<encoder_T<uint32_t>, decoder_T<uint32_t>>("nSliceRowClusters", compressedClusters.nSliceRowClusters, writer);
  writer.EndObject();
};

template <typename source_T>
using sseRansEncoder_t = typename o2::rans::LiteralSIMDEncoder<uint64_t, uint32_t, source_T, 16, 2>;
template <typename source_T>
using sseRansDecoder_t = typename o2::rans::LiteralSIMDDecoder<uint64_t, uint32_t, source_T, 16, 2>;

template <typename source_T>
using avxRansEncoder_t = typename o2::rans::LiteralSIMDEncoder<uint64_t, uint32_t, source_T, 16, 4>;
template <typename source_T>
using avxRansDecoder_t = typename o2::rans::LiteralSIMDDecoder<uint64_t, uint32_t, source_T, 16, 4>;

int main(int argc, char* argv[])
{

  bpo::options_description options("Allowed options");
  // clang-format off
  options.add_options()
    ("help,h", "print usage message")
    ("file,f",bpo::value<std::string>(), "file to compress")
    ("log_severity,l",bpo::value<std::string>(), "severity of FairLogger");
  // clang-format on

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  bpo::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  const std::string filename = [&]() {
    if (vm.count("file")) {
      return vm["file"].as<std::string>();
    } else {
      LOG(error) << "missing path to input file";
      exit(1);
    }
  }();

  if (vm.count("log_severity")) {
    fair::Logger::SetConsoleSeverity(vm["log_severity"].as<std::string>().c_str());
  }

  std::ofstream of{"out.json"};
  if (!of) {
    std::runtime_error(fmt::format("could not open output file at path {}", filename));
  }
  rapidjson::OStreamWrapper stream{of};
  rapidjson::Writer<rapidjson::OStreamWrapper> writer{stream};
  writer.StartObject();

  TPCCompressedClusters compressedClusters = readFile(filename);
  LOG(info) << "loaded Compressed Clusters from file";
  LOG(info) << "######################################################";
  LOG(info) << "start rANS LiteralEncode/Decode";
  encodeTPC<o2::rans::LiteralEncoder64, o2::rans::LiteralDecoder64>("LiteralEncoder64", compressedClusters, false, writer);
  LOG(info) << "######################################################";
  LOG(info) << "start rANS SSE LiteralEncode/Decode";
  encodeTPC<sseRansEncoder_t, sseRansDecoder_t>("SSE16", compressedClusters, false, writer);
  LOG(info) << "######################################################";
  LOG(info) << "start rANS AVX LiteralEncode/Decode";
  encodeTPC<avxRansEncoder_t, avxRansDecoder_t>("AVX16", compressedClusters, false, writer);

  writer.EndObject();
  writer.Flush();
  of.close();
};