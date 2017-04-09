#include "DataCompression/dc_primitives.h"
#include "DataCompression/HuffmanCodec.h"
#include <bitset>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/string.hpp>

/**
 * Parameter model definitions
 * - boost mpl vector of alphabets
 */
typedef boost::mpl::vector<
  BitRangeContiguousAlphabet<uint16_t,  6 , boost::mpl::string < 'p','a','d','r','o','w' >    >,
  BitRangeContiguousAlphabet<uint16_t, 14 , boost::mpl::string < 'p','a','d' >                >,
  BitRangeContiguousAlphabet<uint16_t, 15 , boost::mpl::string < 't','i','m','e' >            >,
  BitRangeContiguousAlphabet<uint16_t,  8 , boost::mpl::string < 's','i','g','m','a','Y','2' >>,
  BitRangeContiguousAlphabet<uint16_t,  8 , boost::mpl::string < 's','i','g','m','a','Z','2' >>,
  BitRangeContiguousAlphabet<uint16_t, 16 , boost::mpl::string < 'c','h','a','r','g','e' >    >,
  BitRangeContiguousAlphabet<uint16_t, 10 , boost::mpl::string < 'q','m','a','x' >            >
  > tpccluster_parameter;
/**
 * Definition of Huffman probability models for the above defined alphabets
 *
 * This is a temporary definition, the mpl sequence can be created automatically
 * from the list of alphabet types, but did not manage so far (see below)
 */
typedef boost::mpl::vector<
  o2::HuffmanModel<ProbabilityModel<BitRangeContiguousAlphabet<uint16_t,  6 , boost::mpl::string < 'p','a','d','r','o','w' >    >>, AliceO2::HuffmanNode<std::bitset<64>>, true>,
  o2::HuffmanModel<ProbabilityModel<BitRangeContiguousAlphabet<uint16_t, 14 , boost::mpl::string < 'p','a','d' >                >>, AliceO2::HuffmanNode<std::bitset<64>>, true>,
  o2::HuffmanModel<ProbabilityModel<BitRangeContiguousAlphabet<uint16_t, 15 , boost::mpl::string < 't','i','m','e' >            >>, AliceO2::HuffmanNode<std::bitset<64>>, true>,
  o2::HuffmanModel<ProbabilityModel<BitRangeContiguousAlphabet<uint16_t,  8 , boost::mpl::string < 's','i','g','m','a','Y','2' >>>, AliceO2::HuffmanNode<std::bitset<64>>, true>,
  o2::HuffmanModel<ProbabilityModel<BitRangeContiguousAlphabet<uint16_t,  8 , boost::mpl::string < 's','i','g','m','a','Z','2' >>>, AliceO2::HuffmanNode<std::bitset<64>>, true>,
  o2::HuffmanModel<ProbabilityModel<BitRangeContiguousAlphabet<uint16_t, 16 , boost::mpl::string < 'c','h','a','r','g','e' >    >>, AliceO2::HuffmanNode<std::bitset<64>>, true>,
  o2::HuffmanModel<ProbabilityModel<BitRangeContiguousAlphabet<uint16_t, 10 , boost::mpl::string < 'q','m','a','x' >            >>, AliceO2::HuffmanNode<std::bitset<64>>, true>
  > tpccluster_parameter_models;

/**
 * this was an attemp to create the vector of Huffman models directly
 * from the vector of alphabets
 *
 * For the moment, the placeholders of mpl fold are not expanded, so there are
 * unknown types in the end
 */
  /// very first attemp
  //using namespace boost::mpl::placeholders;
  //
  //typedef boost::mpl::fold<
  //  tpccluster_parameter,
  //  boost::mpl::vector<>,
  //  boost::mpl::push_back<
  //    _1
  //    , AliceO2::HuffmanModel< ProbabilityModel< _2 >, AliceO2::HuffmanNode<std::bitset<64>>, true>
  //    >
  //  >::type models_t;

  /// trying with additional lambda levels
  //typedef boost::mpl::string<'T','e','s','t'>::type TestAlphabetName;
  //typedef ContiguousAlphabet<int16_t, -1, 10, TestAlphabetName> TestAlphabet;
  //
  //typedef typename boost::mpl::lambda< ProbabilityModel< _1 > > apply_alphabet;
  //typedef boost::mpl::apply1<apply_alphabet, TestAlphabet>::type TestAlphabetModel;
  //typedef typename boost::mpl::lambda< AliceO2::HuffmanModel< _1, AliceO2::HuffmanNode<std::bitset<64>>, true> > apply_probabilitymodel;
  //typedef typename boost::mpl::apply1<boost::mpl::protect<apply_probabilitymodel>::type, TestAlphabetModel>::type TestHuffmanModel;
  //
  //TestAlphabetModel object;
  //typedef TestAlphabetModel::value_type vtype;
  //
  //std::cout << object.getName() << std::endl;


  //typedef boost::mpl::fold<
  //  tpccluster_parameter,
  //  boost::mpl::vector<>,
  //  boost::mpl::push_back<
  //    _1
  //    , boost::mpl::apply1< boost::mpl::protect<apply_huffmanmodel>::type, _2 >
  //    >
  //  >::type models_t;
