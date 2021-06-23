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

/* Local Variables:  */
/* mode: c++         */
/* End:              */

#ifndef HUFFMANCODEC_H
#define HUFFMANCODEC_H

/// @file   HuffmanCodec.h
/// @author Matthias Richter
/// @since  2016-08-11
/// @brief  Implementation of a Huffman codec

#include <cstdint>
#include <cerrno>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream> // stringstream in configuration parsing
#include "CommonUtils/CompStream.h"

namespace o2
{
namespace data_compression
{

/**
 * @class HuffmanNode
 * @brief Container holding information to build Huffman tree
 *
 * The container holds information about child nodes in the tree, the
 * accumulated weight (probability) according to coding model. Leave node, i.e.
 * the end of the tree branches, also contain the Huffman code after assignment.
 */
template <typename _CodeType>
class HuffmanNode
{
 public:
  using self_type = HuffmanNode;
  using pointer = self_type*;
  using shared_pointer = std::shared_ptr<self_type>;
  using CodeType = _CodeType;

  HuffmanNode() : mLeft(), mRight(), mWeight(0.), mIndex(~uint16_t(0)), mCode(), mCodeLen(0) {}
  HuffmanNode(const HuffmanNode& other) = default;
  HuffmanNode& operator=(const HuffmanNode& other) = default;
  ~HuffmanNode() = default;

  HuffmanNode(double weight, uint16_t index = ~uint16_t(0))
    : mLeft(), mRight(), mWeight(weight), mIndex(index), mCode(), mCodeLen(0)
  {
  }

  HuffmanNode(shared_pointer left, shared_pointer right)
    : mLeft(left),
      mRight(right),
      mWeight((mLeft ? mLeft->mWeight : 0.) + (mRight ? mRight->mWeight : 0.)),
      mIndex(~uint16_t(0)),
      mCode(),
      mCodeLen(0)
  {
  }

  bool operator<(const HuffmanNode& other) const { return mWeight < other.mWeight; }

  CodeType getBinaryCode() const { return mCode; }
  uint16_t getBinaryCodeLength() const { return mCodeLen; }
  uint16_t getIndex() const { return mIndex; }

  // TODO: can be combined to one function with templated index
  pointer getLeftChild() const { return mLeft.get(); }
  pointer getRightChild() const { return mRight.get(); }
  void setBinaryCode(uint16_t codeLen, CodeType code)
  {
    mCode = code;
    mCodeLen = codeLen;
  }

  // insert bit at LSB
  self_type& operator<<=(bool bit)
  {
    mCode <<= 1;
    if (bit) {
      mCode.set(0);
    } else {
      mCode.reset(0);
    }
    mCodeLen += 1;
    return *this;
  }

  // insert bit at MSB
  self_type& operator>>=(bool bit)
  {
    if (bit) {
      mCode.set(mCodeLen);
    } else {
      mCode.reset(mCodeLen);
    }
    mCodeLen += 1;
    return *this;
  }

  void print(std::ostream& stream = std::cout) const
  {
    static int level = 1;
    stream << "node weight: " << std::setw(9) << mWeight;
    if (!mLeft) {
      stream << " leave ";
    } else {
      stream << "  tree ";
    }
    stream << " code length: " << mCodeLen;
    stream << " code " << mCode;
    stream << std::endl;
    level++;
    if (mLeft) {
      stream << std::setw(level) << level << ":  left: ";
      mLeft->print(stream);
    }
    if (mRight) {
      stream << std::setw(level) << level << ": right: ";
      mRight->print(stream);
    }
    level--;
  }

 private:
  shared_pointer mLeft;
  shared_pointer mRight;
  double mWeight;
  uint16_t mIndex;
  CodeType mCode;
  uint16_t mCodeLen;
};

/**
 * @class HuffmanCodec
 * @brief Main class of the Huffman codec implementation
 *
 * The codec forwards the encoding/decoding requests to the implementation of
 * the coding model.
 *
 * TODO:
 * - Multi parameter support
 * - Interface methods to methods of the instances of the coding model
 */
template <typename _CodingModel>
class HuffmanCodec
{
 public:
  using model_type = _CodingModel;
  using value_type = typename model_type::value_type;
  using code_type = typename model_type::code_type;

  HuffmanCodec() = delete; // forbidden
  HuffmanCodec(const model_type& model) : mCodingModel(model) {}
  ~HuffmanCodec() = default;

  /// Encode a value and return Huffman code and code lenght
  /// @param v          value to be encoded
  /// @param code       [out] returned code
  /// @param codeLength [out] length of the returned code
  template <typename ValueType>
  const bool Encode(ValueType v, code_type& code, uint16_t& codeLength) const
  {
    code = mCodingModel.Encode(v, codeLength);
    return true;
  }

  /// Decode a value
  /// @param v          [out] decoded value
  /// @param code       [in] code to be processed
  /// @param codeLength [out] length of the processed code
  template <typename ReturnType>
  bool Decode(ReturnType& v, code_type code, uint16_t& codeLength) const
  {
    v = mCodingModel.Decode(code, codeLength);
    return true;
  }

  /// Get the underlying coding model
  model_type const& getCodingModel() const
  {
    return mCodingModel;
  }

  /// Load configuration from file
  ///
  /// Text file can be in compressed format, supported methods: gzip, zlib, bzip2, and lzma
  int loadConfiguration(const char* filename, std::string method = "zlib")
  {
    return mCodingModel.read(filename, method);
  }

  /// Write configuration to file
  ///
  /// Can be written in compressed format, supported methods: gzip, zlib, bzip2, and lzma
  int writeConfiguration(const char* filename, std::string method = "zlib") const
  {
    return mCodingModel.write(filename, method);
  }

 private:
  model_type mCodingModel;
};

/**
 * @class Huffman model
 * @brief Probability model implementing Huffman functionality
 * This is a mixin class which extends the ProbabilityModel base
 *
 * TODO:
 * - Alphabet object: right now only a default object of the alphabet is
 *   supported, all functionality needs to be implemented in the type
 * - assignment loop can be simplified by a specialized implementation of operator<<=,
 *   alternatively a struct combining code value and length and assignment operator=
 * - type traits for code_type to provide set and reset functions for primitive
 *   data types
 * - check max length possible by code_type to be compatible with required length
 * - class StorageType as template parameter for Alphabet type
 * - error policy
 */
template <typename _BASE, typename Rep, bool orderMSB = true>
class HuffmanModel : public _BASE
{
 public:
  HuffmanModel() : mAlphabet(), mLeaveNodes(), mTreeNodes() {}
  ~HuffmanModel() = default;

  using base_type = _BASE;
  using code_type = Rep;
  using node_type = HuffmanNode<code_type>;
  using value_type = typename _BASE::value_type;
  static constexpr bool OrderMSB = orderMSB;

  int init(double v = 1.) { return _BASE::initWeight(mAlphabet, v); }

  /**
   * Encode value
   *
   * @arg symbol     [in]  symbol to be encoded
   * @arg codeLength [OUT] code length, number of LSBs
   * @return Huffman code, valid if codeLength > 0
   */
  code_type Encode(typename _BASE::value_type symbol, uint16_t& codeLength) const
  {
    codeLength = 0;
    auto nodeIndex = _BASE::alphabet_type::getIndex(symbol);
    if (nodeIndex < mLeaveNodes.size()) {
      // valid symbol/value
      codeLength = mLeaveNodes[nodeIndex]->getBinaryCodeLength();
      return mLeaveNodes[nodeIndex]->getBinaryCode();
    } else {
      std::string msg = "symbol ";
      msg += symbol;
      msg += " not found in alphapet ";
      msg += _BASE::getName();
      throw std::range_error(msg);
    }

    static const code_type dummy = 0;
    return dummy;
  }

  /**
   * Decode bit pattern
   *
   * The caller provides a bit field determined by the class' template parameter,
   * The number of decoded bits is indicated in the codeLength parameter after decoding.
   * @arg code        [in]  code bits
   * @arg codeLength  [OUT] number of decoded bits
   * @return value, valid if codeLength > 0
   */
  value_type Decode(code_type code, uint16_t& codeLength) const
  {
    // TODO: need to check if there is a loaded tree, but don't
    // want to check this every time when calling. Maybe its enough
    // to let the dereferencing below throw an exception
    codeLength = 0;
    typename _BASE::value_type v = 0;
    // dereference the iterator and get raw pointer from shared pointer
    // TODO: work on shared pointers here as well
    // the top node is the only element in the multiset after using the
    // weighted sort algorithm to build the tree, all nodes are referenced
    // from their parents in the tree.
    const node_type* node = (*mTreeNodes.begin()).get();
    uint16_t codeMSB = code.size() - 1;
    while (node) {
      // N.B.: nodes have either both child nodes or none of them
      if (node->getLeftChild() == nullptr) {
        // this is a leave node, retrieve value for corresponding index
        // TODO: validity check for index, this can be done once after
        // initializing the Huffman configuration, either after training
        // or loading the configuration
        return _BASE::alphabet_type::getSymbol(node->getIndex());
      }
      if (codeLength > codeMSB) {
        // the size of the code type is shorter than the Huffman tree length
        throw std::range_error("code type length insufficient for Huffman tree length");
        break;
      }
      bool bit = false;
      if (OrderMSB) {
        bit = code.test(codeMSB - codeLength);
      } else {
        bit = code.test(codeLength);
      }
      ++codeLength;
      if (bit) {
        node = node->getLeftChild();
      } else {
        node = node->getRightChild();
      }
    }
    return v;
  }

  /**
   * 'less' functor used in the multiset for sorting in the order less
   * probable to more probable
   */
  template <typename T>
  class isless
  {
   public:
    bool operator()(const T a, const T b) const { return a < b; }
  };
  /// specialization for pointer types
  template <typename T>
  class isless<T*>
  {
   public:
    bool operator()(const T* a, const T* b) const
    {
      if (a == nullptr || b == nullptr) {
        return false;
      }
      return *a < *b;
    }
  };
  /// specialization for shared pointer
  template <typename T>
  class isless<std::shared_ptr<T>>
  {
   public:
    bool operator()(const std::shared_ptr<T>& a, const std::shared_ptr<T>& b) const
    {
      if (!a || !b) {
        return false;
      }
      return *a < *b;
    }
  };

  /**
   * Combine and sort nodes to build a binary tree
   * TODO: separate data structures for tree and leaf nodes to optimize
   * storage
   */
  bool GenerateHuffmanTree()
  {
    mLeaveNodes.clear();

    // probability model provides map of {symbol, weight}-pairs
    _BASE& model = *this;
    for (auto i : model) {
      // create nodes knowing about their index and the symbol weight
      mLeaveNodes.push_back(std::make_shared<node_type>(i.second, _BASE::alphabet_type::getIndex(i.first)));
    }

    // insert pointer to nodes into ordered structure to build tree
    // since the type is a pointer, a specific 'less' functor needs to
    // be provided to dereference before applying operator<
    for (auto& i : mLeaveNodes) {
      mTreeNodes.insert(i);
    }
    while (mTreeNodes.size() > 1) {
      // create new node combining the two with lowest probability
      std::shared_ptr<node_type> combinedNode = std::make_shared<node_type>(*mTreeNodes.begin(), *++mTreeNodes.begin());
      // remove those two nodes from the list
      mTreeNodes.erase(mTreeNodes.begin());
      mTreeNodes.erase(mTreeNodes.begin());
      // insert the new node according to the less functor
      mTreeNodes.insert(combinedNode);
    }
    // assign value, method works on pointer
    // dereference iterator and shared_ptr to get the raw pointer
    // TODO: change method to work on shared instead of raw pointers
    assignCode((*mTreeNodes.begin()).get());
    return true;
  }

  /**
   * assign code to this node loop to right and left nodes
   *
   * Code direction is determined by template parameter orderMSB being either
   * true or false.
   * Code can be built up in two directions, either with the bit of the parent
   * node in the MSB or LSB. In the latter case, the bit of the parent node
   * has to be right of the bit of child nodes, i.e. bits correspond to the
   * current code length. A bit stream storing bits from MSB to LSB and then
   * overwrapping to the MSBs of the next byte, requires to code to start with
   * MSB.
   *
   * TODO: implement iterator concept
   */
  int assignCode(node_type* node)
  {
    if (node == nullptr) {
      return 0;
    }
    int codelen = node->getBinaryCodeLength();
    int retcodelen = codelen;
    if (node->getLeftChild()) { // bit '1' branch
      code_type c = node->getBinaryCode();
      if (OrderMSB) { // note: this is a compile time switch
        c <<= 1;
        c.set(0);
      } else {
        c.set(codelen);
      }
      node->getLeftChild()->setBinaryCode(codelen + 1, c);
      int branchlen = assignCode(node->getLeftChild());
      if (retcodelen < branchlen) {
        retcodelen = branchlen;
      }
    }
    if (node->getRightChild()) { // bit '0' branch
      code_type c = node->getBinaryCode();
      if (OrderMSB) {
        c <<= 1;
        c.reset(0);
      } else {
        c.reset(codelen);
      }
      node->getRightChild()->setBinaryCode(codelen + 1, c);
      int branchlen = assignCode(node->getRightChild());
      if (retcodelen < branchlen) {
        retcodelen = branchlen;
      }
    }
    return retcodelen;
  }

  /**
   * @brief Write Huffman table to file.
   *
   * The file can be compressed on-the-fly, supported methods:
   * gzip, zlib, bzip2, and lzma
   *
   * Configuration is written in self-consistent text file format.
   */
  int write(const char* filename, std::string method = "zlib") const
  {
    o2::io::ocomp_stream out(filename, method);
    return write(out);
  }

  /**
   * @brief Write Huffman table in self-consistent format to an output stream
   */
  int write(std::ostream& out) const
  {
    if (mTreeNodes.size() == 0) {
      return 0;
    }
    return write(out, (*mTreeNodes.begin()).get());
  }

  /**
   * @brief Read configuration from file
   *
   * Read configuration text file, can be in compressed format, supported
   * methods: gzip, zlib, bzip2, and lzma
   */
  int read(const char* filename, std::string method = "zlib")
  {
    o2::io::icomp_stream in(filename, method);
    return read(in);
  }

  /**
   * @brief Read configuration from stream
   * The previous configuration is discarded.
   *
   * The text file contains a self-consistent representatio of the Huffman tree,
   * one node definition per line. Each node has an index (corresponding to the
   * line number). The tree nodes hold the indices of their child nodes.
   * Leave node format: index value weight codelen code
   * Tree node format:  index left_index right_index
   */
  int read(std::istream& in)
  {
    mLeaveNodes.clear();
    mTreeNodes.clear();
    int lineNo = -1;
    std::string node, left, right, parameters;
    std::set<int> nodeIndices;
    struct treeNodeConfiguration {
      treeNodeConfiguration(int _index, int _left, int _right) : index(_index), left(_left), right(_right) {}
      int index, left, right;
      bool operator<(const treeNodeConfiguration& other) const { return index < other.index; }
    };
    std::set<treeNodeConfiguration> treeNodeConfigurations;
    char firstChar = 0;
    std::map<int, int> leaveNodeMap;
    while ((in.get(firstChar)) && firstChar != '\n' && (in.putback(firstChar)) && (in >> node) && (in >> left) &&
           (in >> right) && ++lineNo >= 0) {
      std::getline(in, parameters);
      if (lineNo != std::stoi(node)) {
        std::cerr << "format error: expected node no " << lineNo << ", but got " << node << " (" << left << " " << right
                  << " " << parameters << ")" << std::endl;
        std::cerr << "Note: Huffman table dump has to be terminated by blank line or eof" << std::endl;
        break;
      }
      if (parameters.empty()) {
        // std::cout << "tree node " << lineNo << " left=" << left << " right=" << right << std::endl;
        int leftIndex = std::stoi(left);
        int rightIndex = std::stoi(right);
        auto it = nodeIndices.find(leftIndex);
        if (it == nodeIndices.end()) {
          std::cerr << "Format error: can not find left child node with index " << leftIndex << std::endl;
          return -1;
        }
        nodeIndices.erase(it);
        it = nodeIndices.find(rightIndex);
        if (it == nodeIndices.end()) {
          std::cerr << "Format error: can not find right child node with index " << rightIndex << std::endl;
          return -1;
        }
        nodeIndices.erase(it);
        treeNodeConfigurations.insert(treeNodeConfiguration(lineNo, leftIndex, rightIndex));
      } else {
        std::stringstream vs(left), ws(right);
        typename _BASE::alphabet_type::value_type symbol;
        vs >> symbol;
        typename _BASE::weight_type weight;
        ws >> weight;
        int symbolIndex = _BASE::alphabet_type::getIndex(symbol);
        // grow the vector as operator[] always expects index within range
        if (mLeaveNodes.size() < symbolIndex + 1) {
          mLeaveNodes.resize(symbolIndex + 1);
        }
        mLeaveNodes[symbolIndex] = std::make_shared<node_type>(weight, symbolIndex);
        std::stringstream ps(parameters);
        uint16_t codeLen = 0;
        ps >> codeLen;
        code_type code = 0;
        ps >> code;
        mLeaveNodes[symbolIndex]->setBinaryCode(codeLen, code);
        leaveNodeMap[lineNo] = symbolIndex;
        _BASE::addWeight(symbol, weight);
        // std::cout << "leave node " << lineNo << " " << " value=" << value << " weight=" << weight << " " << codeLen
        // << " " << code << std::endl;
      }
      nodeIndices.insert(lineNo);
    }
    std::map<int, std::shared_ptr<node_type>> treeNodes;
    for (auto conf : treeNodeConfigurations) {
      std::shared_ptr<node_type> left_local;
      auto ln = leaveNodeMap.find(conf.left);
      if (ln != leaveNodeMap.end()) {
        left_local = mLeaveNodes[ln->second];
        leaveNodeMap.erase(ln);
      } else {
        auto tn = treeNodes.find(conf.left);
        if (tn == treeNodes.end()) {
          std::cerr << "Internal error: can not find left child node with index " << conf.left << std::endl;
          return -1;
        }
        left_local = tn->second;
        treeNodes.erase(tn);
      }
      std::shared_ptr<node_type> right_local;
      auto rn = leaveNodeMap.find(conf.right);
      if (rn != leaveNodeMap.end()) {
        right_local = mLeaveNodes[rn->second];
        leaveNodeMap.erase(rn);
      } else {
        auto tn = treeNodes.find(conf.right);
        if (tn == treeNodes.end()) {
          std::cerr << "Internal error: can not find right child node with index " << conf.right << std::endl;
          return -1;
        }
        right_local = tn->second;
        treeNodes.erase(tn);
      }
      // make combined node shared ptr and add to map
      treeNodes[conf.index] = std::make_shared<node_type>(left_local, right_local);
    }
    if (leaveNodeMap.size() != 0 || treeNodes.size() != 1) {
      std::cerr << "error: " << leaveNodeMap.size() << " unhandled leave node(s)"
                << "; " << treeNodes.size() << " tree nodes(s), expected 1" << std::endl;
    }
    mTreeNodes.insert(treeNodes.begin()->second);
    return 0;
  }

  void print() const
  {
    if (mTreeNodes.size() > 0) {
      node_type* topNode = (*mTreeNodes.begin()).get();
      if (topNode) {
        topNode->print();
      } else {
        // TODO: handle this error condition
      }
    }
  };

 private:
  /**
   * @brief Recursive write of the node content.
   *
   * Iterate through Huffman tree and write information first of the
   * leave of each branch and then the corresponding parent tree nodes.
   */
  template <typename NodeType>
  int write(std::ostream& out, NodeType* node, int nodeIndex = 0) const
  {
    if (!node) {
      return nodeIndex;
    }
    const _BASE& model = *this;
    NodeType* left = node->getLeftChild();
    NodeType* right = node->getRightChild();
    if (left == nullptr) {
      typename _BASE::value_type value = _BASE::alphabet_type::getSymbol(node->getIndex());
      out << nodeIndex << " " << value << " " << model[value] << " " << node->getBinaryCodeLength() << " "
          << node->getBinaryCode() << std::endl;
      return nodeIndex;
    }
    int leftIndex = write(out, left, nodeIndex);
    int rightIndex = write(out, right, leftIndex + 1);
    out << rightIndex + 1 << " " << leftIndex << " " << rightIndex << std::endl;
    return rightIndex + 1;
  }

  // the alphabet, determined by template parameter
  typename _BASE::alphabet_type mAlphabet;
  // Huffman leave nodes containing symbol index to code mapping
  std::vector<std::shared_ptr<node_type>> mLeaveNodes;
  // multiset, order determined by less functor working on pointers
  std::multiset<std::shared_ptr<node_type>, isless<std::shared_ptr<node_type>>> mTreeNodes;
};

} // namespace data_compression
} // namespace o2

#endif
