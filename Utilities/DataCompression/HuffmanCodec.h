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

namespace AliceO2 {

/** TODO:
    - ensure that _ValueType is not a float type
    - max value at compile time depending on bit width of _CodeType
    - generalization of the alphabet to be encoded, e.g. alphabet policy with
      an iterator of the valid elements of the alphabet. This would then
      also allow symbol ranges other then 0 to some max value

    - class StorageType as template parameter for Alphabet type
    - code direction can be a property of the node, assignment loop can be
      simplified by a specialized implementation of operator<<=, alternatively
      a struct combining code value and length and assignment operator=
    - right now, the code direction in assignCode is fixed by the default of
      the order template argument
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

template<
  class    _AlphabetType,
  typename _NodeType
  >
class HuffmanCodec {
 public:
  HuffmanCodec() : mAlphabet(), mLeaveNodes(), mLeaveNodeIndex(), mValueIndex(), mTreeNodes() {}
  ~HuffmanCodec() {}

  typedef typename _NodeType::CodeType _CodeType;

  /// Return huffman code for a value
  const _CodeType Encode(typename _AlphabetType::value_type v, uint16_t& codeLength) const {
    codeLength = 0;
    auto nodeIndex = mLeaveNodeIndex.find(v);
    if (nodeIndex != mLeaveNodeIndex.end() && nodeIndex->second < mLeaveNodes.size()) {
      // valid symbol/value
      codeLength = mLeaveNodes[nodeIndex->second].getBinaryCodeLength();
      return mLeaveNodes[nodeIndex->second].getBinaryCode();
    } else {
      // TODO: error policy: invalid value
    }

    static const _CodeType dummy = 0;
    return dummy;
  }

  template<bool orderMSB = true>
  typename _AlphabetType::value_type Decode(_CodeType code, uint16_t& codeLength) const {
    codeLength = 0;
    typename _AlphabetType::value_type v = 0;
    const _NodeType* node = *mTreeNodes.begin();
    uint16_t codeMSB = code.size() - 1;
    while (node) {
      if (node->getLeftChild() == nullptr) {
	// this is a leave node
	if (mValueIndex.find(node->getIndex()) != mValueIndex.end()) {
	  return mValueIndex.find(node->getIndex())->second;
	}
	// something wrong here
	break;
      }
      if (codeLength > codeMSB) {
	// something wrong here
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
  template<
    class ProbabilityModel
    >
  bool GenerateHuffmanTree(ProbabilityModel model) {
    mLeaveNodes.clear();
    mLeaveNodeIndex.clear();
    // insert pointer to nodes into ordered structure to build tree
    // since the type is a pointer, a specific 'less' functor needs to
    // be provided to dereference before applying operator<
    uint16_t index = 0;
    for ( auto i : model) {
      mLeaveNodes.push_back(_NodeType(i.second, index));
      mLeaveNodeIndex[i.first] = index;
      mValueIndex[index++] = i.first;
    }
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
    _NodeType* topNode = *mTreeNodes.begin();
    //assign value
    assignCode(topNode);
    topNode->print();
    return true;
  }

  /**
     TODO: implement iterator concept
   */
  template<
    class NodeType = _NodeType,
    bool  OrderMSB = true
    >
  int assignCode(NodeType* node) {
    /// assign code to this node loop to right and left nodes
    /// code can be built up in two directions, either with the bit of the parent
    /// node in the MSB or LSB. In the latter case, the bit of the parent node
    /// has to be right of the bit of child nodes, i.e. bits correspond to the
    /// current code length. A bit stream storing bits from MSB to LSB and then
    /// overwrapping to the MSBs of the next byte, requires to code to start with
    /// MSB.
    /// TODO: type traits for _CodeType to povide set and reset functions for
    /// primitive data types
    if (node == nullptr) return 0;
    int codelen = node->getBinaryCodeLength();
    int retcodelen = codelen;
    if (node->getLeftChild()) {// bit '1' branch
      _CodeType c = node->getBinaryCode();
      if (OrderMSB) {// note: this is a compile time switch
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
      _CodeType c = node->getBinaryCode();
      if (OrderMSB) {
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

 private:
  _AlphabetType* mAlphabet;
  std::vector<_NodeType> mLeaveNodes;
  std::map<typename _AlphabetType::value_type, uint16_t > mLeaveNodeIndex;
  std::map<uint16_t , typename _AlphabetType::value_type> mValueIndex;
  std::multiset<_NodeType*, pless<_NodeType*> > mTreeNodes;

};
};

#endif
