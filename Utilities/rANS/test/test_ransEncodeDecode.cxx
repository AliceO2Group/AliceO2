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

template <typename CODER_T, typename STREAM_T, uint P>
struct Fixture {
  // type of the coder: 32 bit or 64 bit
  using coder_t = CODER_T;
  // how many bytes do we stream out during normalization:
  // 1 Byte for 32Bit coders, 4Byte for 64Bit coders
  using stream_t = STREAM_T;
  // what is the datatype of our source symbols?
  using source_t = char;

  using encoder_t = o2::rans::Encoder<coder_t, stream_t, source_t>;
  using decoder_t = o2::rans::Decoder<coder_t, stream_t, source_t>;

  //TUNIG parameters
  // how many bits do we resample the symbol statistics to?
  // this depends on the size of the alphabet (i.e. how big source_t is).
  // See poster for more details
  // https://indico.cern.ch/event/773049/contributions/3474364/attachments/1936180/3208584/Layout.pdf
  // As a rule of thumb we need
  // 10 Bits for 8Bit alphabets,
  // 22Bits for 16Bit alphabets,
  // 25Bits for 25Bit alphabets
  const uint probabilityBits = P;

  const std::string source = "";
};

template <typename CODER_T, typename STREAM_T, uint P>
struct FixtureFull : public Fixture<CODER_T, STREAM_T, P> {
  const std::string source = R"(Sed ut perspiciatis, unde omnis iste natus error sit voluptatem accusantium
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
};

typedef boost::mpl::vector<Fixture<uint32_t, uint8_t, 14>, Fixture<uint64_t, uint32_t, 18>, FixtureFull<uint32_t, uint8_t, 14>, FixtureFull<uint64_t, uint32_t, 18>> Fixtures;

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_EncodeDecode, T, Fixtures, T)
{

  // iterate over the message and create PDF and CDF for each symbol in the message
  o2::rans::SymbolStatistics stats{std::begin(T::source), std::end(T::source)};
  // For performance reason the distributions must be rescaled to be a power of two.
  stats.rescaleToNBits(T::probabilityBits);

  // buffer to write the rANS coded message into
  std::vector<typename T::stream_t> encoderBuffer(1 << 20, 0);
  //create a stateful encoder object that builds an encoding table from the given symbol statistics
  const typename T::encoder_t encoder{stats, T::probabilityBits};
  // encoder rANS encodes symbols from SOURCE array to encoder Buffer. Since coder and decoder
  // are mathematical inverse functions on the ANS state, they operate as a stack -
  // i.e. the decoder outputs the message in reverse order of the encoder.
  // By convention the encoder runs backwards over the input so that the decoder can return
  // the data in the expected order. The encoder will start from encoderBuffer.begin()
  // and return an iterator one element past the last entry written.
  // This means the encodeded message ranges from encoderBuffer.begin() to encodedMessageEnd -1, with (encodedMessageEnd -encoderBuffer.begin()) entries written.
  auto encodedMessageEnd = encoder.process(encoderBuffer.begin(), encoderBuffer.end(), std::begin(T::source), std::end(T::source));

  // The decoded message will go into the decoder buffer which will have as many symbols as the original message
  std::vector<typename T::source_t> decoderBuffer(std::distance(std::begin(T::source), std::end(T::source)), 0);
  // create a stateful decoder object that build a decoding table from the given symbol statistics
  const typename T::decoder_t decoder{stats, T::probabilityBits};
  // the decoder unwinds the rANS state in the encoder buffer starting at ransBegin and decodes it into the decode buffer;
  decoder.process(decoderBuffer.begin(), encodedMessageEnd, stats.getMessageLength());

  //the decodeBuffer and the source message have to be identical afterwards.
  BOOST_REQUIRE(std::memcmp(&(*T::source.begin()), decoderBuffer.data(),
                            decoderBuffer.size() * sizeof(typename T::source_t)) == 0);
}
