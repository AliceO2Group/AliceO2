//-*- Mode: C++ -*-

#ifndef HUFFMANCODEC_H
#define HUFFMANCODEC_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   HuffmanCodec.h
//  @author Matthias Richter
//  @since  2015-08-11
//  @brief  Implementation of a Huffman codec

#include <cstdint>
#include <cerrno>
#include <set>
#include <map>
#include <exception>
#include <stdexcept>

namespace AliceO2 {

/**
 * @class HuffmanNode
 * @brief Container holding information to build Huffman tree
 *
 * The container holds information about child nodes in the tree, the
 * accumulated weight (probability) according to coding model. Leave node, i.e.
 * the end of the tree branches, also contain the Huffman code after assignment.
 */
template<typename _CodeType>
class HuffmanNode {
public:
  typedef HuffmanNode self_type;
  typedef HuffmanNode* pointer;
  typedef _CodeType CodeType;

  HuffmanNode() : mLeft(nullptr), mRight(nullptr), mWeight(0.), mIndex(~uint16_t(0)), mCode(), mCodeLen(0) {}
  HuffmanNode(const HuffmanNode& other) : mLeft(other.mLeft), mRight(other.mRight), mWeight(other.mWeight), mCode(other.mCode), mIndex(other.mIndex), mCodeLen(other.mCodeLen) {}
  HuffmanNode& operator=(const HuffmanNode& other) {
    if (this != &other) new (this) HuffmanNode(other);
    return *this;
  }
  ~HuffmanNode() {}

  HuffmanNode(double weight, uint16_t index = ~uint16_t(0)) : mLeft(nullptr), mRight(nullptr), mWeight(weight), mIndex(index), mCode(), mCodeLen(0) {}
  HuffmanNode(pointer left, pointer right) : mLeft(left), mRight(right), mWeight((mLeft!=nullptr?mLeft->mWeight:0.)+(mRight!=nullptr?mRight->mWeight:0.)), mIndex(~uint16_t(0)), mCode(), mCodeLen(0) {}

  bool operator<(const HuffmanNode& other) const {
    return mWeight < other.mWeight;
  }

  CodeType getBinaryCode() const {return mCode;}
  uint16_t getBinaryCodeLength() const {return mCodeLen;}
  uint16_t getIndex() const {return mIndex;}

  // TODO: can be combined to one function with templated index
  pointer  getLeftChild() const {return mLeft;}
  pointer  getRightChild() const {return mRight;}
  void setBinaryCode(uint16_t codeLen, CodeType code) {mCode = code; mCodeLen = codeLen;}

  self_type& operator<<=(bool bit) {
    mCode <<= 1;
    if (bit) mCode.set(0);
    else mCode.reset(0);
    mCodeLen +=1;
    return *this;
  }

  self_type& operator>>=(bool bit) {
    if (bit) mCode.set(mCodeLen);
    else mCode.reset(mCodeLen);
    mCodeLen +=1;
    return *this;
  }

  void print(std::ostream& stream = std::cout) const {
    static int level=1;
    stream << "node weight: " << std::setw(9) << mWeight;
    if (mLeft == nullptr) stream << " leave ";
    else                  stream << "  tree ";
    stream << " code length: " << mCodeLen;
    stream << " code " << mCode;
    stream << std::endl;
    level++;
    if (mLeft!=nullptr)  {stream << std::setw(level) << level << ":  left: "; mLeft->print(stream);}
    if (mRight!=nullptr) {stream << std::setw(level) << level << ": right: "; mRight->print(stream);}
    level--;
  }

private:
  pointer mLeft;
  pointer mRight;
  double  mWeight;
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
template<
  typename _CodingModel
  >
class HuffmanCodec {
 public:
  HuffmanCodec(const _CodingModel& model) : mCodingModel(model) {}
  ~HuffmanCodec() {}

  /// Return Huffman code for a value
  template<typename CodeType, typename ValueType>
  const bool Encode(ValueType v, CodeType& code, uint16_t& codeLength) const {
    code = mCodingModel.Encode(v, codeLength);
    return true;
  }

  template<typename ReturnType, typename CodeType, bool orderMSB = true>
  bool Decode(ReturnType& v, CodeType code, uint16_t& codeLength) const {
    v = mCodingModel.Decode(code, codeLength);
    return true;
  }

 private:
  HuffmanCodec(); //forbidden
  _CodingModel mCodingModel;
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
template<typename _BASE, typename _NodeType, bool _orderMSB = true>
class HuffmanModel : public _BASE {
public:
  HuffmanModel() : mAlphabet(), mLeaveNodes(), mLeaveNodeIndex(), mValueIndex(), mTreeNodes() {}
  ~HuffmanModel() {}

  typedef _BASE                                base_type;
  typedef class HuffmanModel<_BASE, _NodeType> self_type;
  typedef typename _BASE::value_type value_type;
  typedef typename _NodeType::CodeType code_type;
  static constexpr bool orderMSB = _orderMSB;

  int init(double v = 1.) {return _BASE::initWeight(mAlphabet, v);}

  /**
   * Encode value
   *
   * @arg symbol     [in]  symbol to be encoded
   * @arg codeLength [OUT] code length, number of LSBs
   * @return Huffman code, valid if codeLength > 0
   */
  code_type Encode(typename _BASE::value_type symbol, uint16_t& codeLength) const {
    codeLength = 0;
    auto nodeIndex = mLeaveNodeIndex.find(symbol);
    if (nodeIndex != mLeaveNodeIndex.end() && nodeIndex->second < mLeaveNodes.size()) {
      // valid symbol/value
      codeLength = mLeaveNodes[nodeIndex->second].getBinaryCodeLength();
      return mLeaveNodes[nodeIndex->second].getBinaryCode();
    } else {
      std::string msg = "symbol "; msg += symbol;
      msg += " not found in alphapet "; msg += _BASE::getName();
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
  value_type Decode(code_type code, uint16_t& codeLength) const {
    codeLength = 0;
    typename _BASE::value_type v = 0;
    const _NodeType* node = *mTreeNodes.begin();
    uint16_t codeMSB = code.size() - 1;
    while (node) {
      // N.B.: nodes have either both child nodes or none of them
      if (node->getLeftChild() == nullptr) {
        // this is a leave node, find the value for the corresponding
        // index in the value map
        if (mValueIndex.find(node->getIndex()) != mValueIndex.end()) {
          return mValueIndex.find(node->getIndex())->second;
        }
        // something wrong here
        throw std::range_error("the index is not known in the map of values");
        break;
      }
      if (codeLength > codeMSB) {
        // the size of the code type is shorter than the Huffman tree length
        throw std::range_error("code type length insufficient for Huffman tree length");
        break;
      }
      bool bit = false;
      if (orderMSB) bit = code.test(codeMSB - codeLength);
      else bit = code.test(codeLength);
      ++codeLength;
      if (bit) node = node->getLeftChild();
      else node = node->getRightChild();
    }
    return v;
  }

  /**
   * 'less' functor for pointer type arguments
   * used in the multiset for sorting in the order less probable to
   * more probable
   */
  template<typename T>
  class pless {
  public:
    bool operator()(const T a, const T b) {
      if (a == nullptr || b == nullptr) return false;
      return *a < *b;
    }
  };

  /**
   * Combine and sort nodes to build a binary tree
   * TODO: separate data structures for tree and leaf nodes to optimize
   * storage
   */
  bool GenerateHuffmanTree() {
    mLeaveNodes.clear();
    mLeaveNodeIndex.clear();

    // probability model provides map of {symbol, weight}-pairs
    uint16_t index = 0;
    _BASE& model = *this;
    for ( auto i : model) {
      // create nodes knowing about their index and the symbol weight
      mLeaveNodes.push_back(_NodeType(i.second, index));
      // map of {symbol, index}-pairs
      mLeaveNodeIndex[i.first] = index;
      // map of {index, symbol}-pairs
      mValueIndex[index++] = i.first;
    }

    // insert pointer to nodes into ordered structure to build tree
    // since the type is a pointer, a specific 'less' functor needs to
    // be provided to dereference before applying operator<
    for ( auto &i : mLeaveNodes) {
      mTreeNodes.insert(&(i));
    }
    while (mTreeNodes.size() > 1) {
      // create new node combining the two with lowest probability
      _NodeType* combinedNode=new _NodeType(*mTreeNodes.begin(), *++mTreeNodes.begin());
      // remove those two nodes from the list
      mTreeNodes.erase(mTreeNodes.begin());
      mTreeNodes.erase(mTreeNodes.begin());
      // insert the new node according to the less functor
      mTreeNodes.insert(combinedNode);
    }
    //assign value
    assignCode(*mTreeNodes.begin());
    return true;
  }

  /**
   * assign code to this node loop to right and left nodes
   *
   * Code direction is determined by template parameter _orderMSB being either
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
  int assignCode(_NodeType* node) {
    if (node == nullptr) return 0;
    int codelen = node->getBinaryCodeLength();
    int retcodelen = codelen;
    if (node->getLeftChild()) {// bit '1' branch
      code_type c = node->getBinaryCode();
      if (orderMSB) {// note: this is a compile time switch
        c <<= 1;
        c.set(0);
      } else {
        c.set(codelen);
      }
      node->getLeftChild()->setBinaryCode(codelen+1, c);
      int branchlen = assignCode(node->getLeftChild());
      if (retcodelen < branchlen)
        retcodelen = branchlen;
    }
    if (node->getRightChild()) {// bit '0' branch
      code_type c = node->getBinaryCode();
      if (orderMSB) {
        c<<=1;
        c.reset(0);
      } else {
        c.reset(codelen);
      }
      node->getRightChild()->setBinaryCode(codelen+1, c);
      int branchlen = assignCode(node->getRightChild());
      if (retcodelen < branchlen)
        retcodelen = branchlen;
    }
    return retcodelen;
  }

  void print() const {
    if (mTreeNodes.size() > 0) {
      _NodeType* topNode = *mTreeNodes.begin();
      topNode->print();
    }
  };

private:
  // the alphabet, determined by template parameter
  typename _BASE::alphabet_type mAlphabet;
  // Huffman leave nodes containing symbol index to code mapping
  std::vector<_NodeType> mLeaveNodes;
  // map of {symbol, index}-pairs
  std::map<typename _BASE::value_type, uint16_t > mLeaveNodeIndex;
  // map of {index, symbol}-patrs
  std::map<uint16_t , typename _BASE::value_type> mValueIndex;
  // multiset, order determined by less functor working on pointers
  std::multiset<_NodeType*, pless<_NodeType*> > mTreeNodes;
};
}; // namespace AliceO2

#endif
