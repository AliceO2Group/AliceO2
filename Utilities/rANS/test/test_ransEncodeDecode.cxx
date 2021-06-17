// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DecoderSymbol.h
/// @author Michael Lettrich
/// @since  2020-04-15
/// @brief  Test rANS encoder/ decoder

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <vector>
#include <cstring>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/vector.hpp>

#include "rANS/rans.h"

struct EmptyTestString {
  std::string data{};
};

struct FullTestString : public EmptyTestString {
  FullTestString()
  {
    data = R"(Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium
doloremque laudantium, totam rem aperiam eaque ipsa, quae ab illo inventore veritatis
et quasi architecto beatae vitae dicta sunt, explicabo. nemo enim ipsam voluptatem,
quia voluptas sit, aspernatur aut odit aut fugit, sed quia consequuntur magni dolores
eos, qui ratione voluptatem sequi nesciunt, neque porro quisquam est, qui dolorem ipsum,
quia dolor sit, amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora
incidunt, ut labore et dolore magnam aliquam quaerat voluptatem. ut enim ad minima veniam,
quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea
commodi consequatur? quis autem vel eum iure reprehenderit, qui in ea voluptate velit
esse, quam nihil molestiae consequatur, vel illum, qui dolorem eum fugiat,
quo voluptas nulla pariatur?)";
  }
};

template <typename coder_T>
struct Params {
};

template <>
struct Params<uint32_t> {
  using coder_t = uint32_t;
  using stream_t = uint8_t;
  using source_t = char;
  static constexpr size_t symbolTablePrecission = 16;
};

template <>
struct Params<uint64_t> {
  using coder_t = uint64_t;
  using stream_t = uint32_t;
  using source_t = char;
  static constexpr size_t symbolTablePrecission = 16;
};

template <
  template <typename, typename, typename> class encoder_T,
  template <typename, typename, typename> class decoder_T,
  typename coder_T, class dictString_T, class testString_T>
struct EncodeDecodeBase {
 public:
  using params_t = Params<coder_T>;

  EncodeDecodeBase()
  {
    dictString_T source;
    std::string& s = source.data;
    o2::rans::FrequencyTable frequencies;
    frequencies.addSamples(std::begin(s), std::end(s));

    encoder = decltype(encoder)(frequencies, params_t::symbolTablePrecission);
    decoder = decltype(decoder)(frequencies, params_t::symbolTablePrecission);

    const auto [min, max] = [&s]() {
      const auto [minIter, maxIter] = std::minmax_element(s.begin(), s.end());
      const char min = minIter == s.end() ? 0 : *minIter;
      const char max = maxIter == s.end() ? 0 : *maxIter + 1;
      return std::make_tuple(min, max);
    }();

    const size_t alphabetRangeBits = o2::rans::internal::numBitsForNSymbols(max - min + 1 + 1);

    BOOST_CHECK_EQUAL(encoder.getSymbolTablePrecision(), params_t::symbolTablePrecission);
    BOOST_CHECK_EQUAL(encoder.getAlphabetRangeBits(), alphabetRangeBits);
    BOOST_CHECK_EQUAL(encoder.getMinSymbol(), min);
    BOOST_CHECK_EQUAL(encoder.getMaxSymbol(), max);

    BOOST_CHECK_EQUAL(decoder.getSymbolTablePrecision(), params_t::symbolTablePrecission);
    BOOST_CHECK_EQUAL(decoder.getAlphabetRangeBits(), alphabetRangeBits);
    BOOST_CHECK_EQUAL(decoder.getMinSymbol(), min);
    BOOST_CHECK_EQUAL(decoder.getMaxSymbol(), max);
  }

  virtual void encode() = 0;
  virtual void decode() = 0;

  void check()
  {
    testString_T testString;
    BOOST_CHECK_EQUAL_COLLECTIONS(testString.data.begin(), testString.data.end(), decodeBuffer.begin(), decodeBuffer.end());
  }

  testString_T source;
  encoder_T<typename params_t::coder_t, typename params_t::stream_t, typename params_t::source_t> encoder{};
  decoder_T<typename params_t::coder_t, typename params_t::stream_t, typename params_t::source_t> decoder{};
  std::vector<typename Params<coder_T>::stream_t> encodeBuffer{};
  std::vector<typename Params<coder_T>::source_t> decodeBuffer{};
};

template <typename coder_T, class dictString_T, class testString_T>
struct EncodeDecode : public EncodeDecodeBase<o2::rans::Encoder, o2::rans::Decoder, coder_T, dictString_T, testString_T> {
  void encode() override
  {
    BOOST_CHECK_NO_THROW(this->encoder.process(std::begin(this->source.data), std::end(this->source.data), std::back_inserter(this->encodeBuffer)));
  };
  void decode() override
  {
    BOOST_CHECK_NO_THROW(this->decoder.process(this->encodeBuffer.end(), std::back_inserter(this->decodeBuffer), this->source.data.size()));
  };
};

template <typename coder_T, class dictString_T, class testString_T>
struct EncodeDecodeLiteral : public EncodeDecodeBase<o2::rans::LiteralEncoder, o2::rans::LiteralDecoder, coder_T, dictString_T, testString_T> {
  void encode() override
  {
    BOOST_CHECK_NO_THROW(this->encoder.process(std::begin(this->source.data), std::end(this->source.data), std::back_inserter(this->encodeBuffer), literals));
  };
  void decode() override
  {
    BOOST_CHECK_NO_THROW(this->decoder.process(this->encodeBuffer.end(), std::back_inserter(this->decodeBuffer), this->source.data.size(), literals));
    BOOST_CHECK(literals.empty());
  };

  std::vector<typename Params<coder_T>::source_t> literals;
};

template <typename coder_T, class dictString_T, class testString_T>
struct EncodeDecodeDedup : public EncodeDecodeBase<o2::rans::DedupEncoder, o2::rans::DedupDecoder, coder_T, dictString_T, testString_T> {
  void encode() override
  {
    BOOST_CHECK_NO_THROW(this->encoder.process(std::begin(this->source.data), std::end(this->source.data), std::back_inserter(this->encodeBuffer), duplicates));
  };
  void decode() override
  {
    BOOST_CHECK_NO_THROW(this->decoder.process(this->encodeBuffer.end(), std::back_inserter(this->decodeBuffer), this->source.data.size(), duplicates));
  };

  using params_t = Params<coder_T>;
  typename o2::rans::DedupEncoder<typename params_t::coder_t,
                                  typename params_t::stream_t,
                                  typename params_t::source_t>::duplicatesMap_t duplicates;
};

using testCase_t = boost::mpl::vector<EncodeDecode<uint32_t, EmptyTestString, EmptyTestString>,
                                      EncodeDecode<uint64_t, EmptyTestString, EmptyTestString>,
                                      EncodeDecode<uint32_t, FullTestString, FullTestString>,
                                      EncodeDecode<uint64_t, FullTestString, FullTestString>,
                                      EncodeDecodeLiteral<uint32_t, EmptyTestString, EmptyTestString>,
                                      EncodeDecodeLiteral<uint64_t, EmptyTestString, EmptyTestString>,
                                      EncodeDecodeLiteral<uint32_t, FullTestString, FullTestString>,
                                      EncodeDecodeLiteral<uint64_t, FullTestString, FullTestString>,
                                      EncodeDecodeLiteral<uint32_t, EmptyTestString, FullTestString>,
                                      EncodeDecodeLiteral<uint64_t, EmptyTestString, FullTestString>,
                                      EncodeDecodeDedup<uint32_t, EmptyTestString, EmptyTestString>,
                                      EncodeDecodeDedup<uint64_t, EmptyTestString, EmptyTestString>,
                                      EncodeDecodeDedup<uint32_t, FullTestString, FullTestString>,
                                      EncodeDecodeDedup<uint64_t, FullTestString, FullTestString>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_encodeDecode, testCase_T, testCase_t)
{
  testCase_T testCase;
  testCase.encode();
  testCase.decode();
  testCase.check();
};