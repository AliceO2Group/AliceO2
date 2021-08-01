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

/// @file   DigitsParser.h
/// @brief  TRD raw data parser for digits

#include "TRDReconstruction/DigitsParser.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/CompressedDigit.h"
#include "DataFormatsTRD/Digit.h"

#include "fairlogger/Logger.h"

//TODO come back and figure which of below headers I actually need.
#include <cstring> //memcpy
#include <string>
#include <vector>
#include <array>
#include <iterator>
#include <bitset>
#include <iostream>

namespace o2::trd
{

inline void DigitsParser::swapByteOrder(unsigned int& word)
{
  word = (word >> 24) |
         ((word << 8) & 0x00FF0000) |
         ((word >> 8) & 0x0000FF00) |
         (word << 24);
}

int DigitsParser::Parse(bool verbose)
{

    auto timedigitparsestart = std::chrono::high_resolution_clock::now(); // measure total processing time
  //we are handed the buffer payload of an rdh and need to parse its contents.
  //producing a vector of digits.
  mVerbose = verbose;
  //  mDataVerbose = true;
  //  mHeaderVerbose = true;

  mState = StateDigitHCHeader;
  mDataWordsParsed = 0; // count of data wordsin data that have been parsed in current call to parse.
  mDigitsFound = 0;     // tracklets found in the data block, mostly used for debugging.
  mBufferLocation = 0;
  mPaddingWordsCounter = 0;
  if (mVerbose) {
    LOG(info) << "Digit Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at pos " << mStartParse;
    if (mByteOrderFix) {

      LOG(info) << " we will not be byte swapping";
    } else {

      LOG(info) << " we will be byte swapping";
    }
  }
  if (mDataVerbose) {
    LOG(info) << "trackletdata to parse begin";
    std::vector<uint32_t> datacopy(mStartParse, mEndParse);
    if (mByteOrderFix) {
      for (auto a : datacopy) {
        swapByteOrder(a);
      }
    }

    LOG(info) << "digitdata to parse with size of " << datacopy.size();
    int loopsize = 0;
    if (datacopy.size() > 1024) {
      loopsize = 64;
    }
    for (int i = 0; i < loopsize; i += 8) {
      LOG(info) << std::hex << "0x" << datacopy[i] << " " << std::hex << "0x" << datacopy[i + 1] << " " << std::hex << "0x" << datacopy[i + 2] << " " << std::hex << "0x" << datacopy[i + 3] << " " << std::hex << "0x" << datacopy[i + 4] << " " << std::hex << "0x" << datacopy[i + 5] << " " << std::hex << "0x" << datacopy[i + 6] << " " << std::hex << "0x" << datacopy[i + 7];
    }
    LOG(info) << "digitdata to parse end";
    if (datacopy.size() > 1024) {
      LOG(error) << "something likely very wrong with digit parsing >1024";
    }
  }
  int mcmdatacount = 0;
  int mcmadccount = 0;
  int digitwordcount = 9;
  int digittimebinoffset = 0;
  if (mVerbose) {
    LOG(info) << "Digits Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at " << std::distance(mData->begin(), mStartParse) << " ending at " << std::distance(mData->begin(), mEndParse);
    LOG(info) << "Digits Parser parse " << std::distance(mStartParse, mEndParse) << " items for digits should be 21*11+header";
    LOG(info) << "word to parse : " << std::hex << *mStartParse << "and " << *(mStartParse + 1) << " in state :" << mState;
  }
  //mData holds a buffer containing digits parse placing the read digits where they need to be
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  // data starts with a DigitHCHeader, so pull that off first to simplify looping
  if (mState == StateDigitHCHeader) {
    if (mVerbose) {
      LOG(info) << "at start of data";
    }
    mDigitHCHeader = (DigitHCHeader*)mStartParse;
    if (mByteOrderFix) {
      // byte swap if needed.
      swapByteOrder(mDigitHCHeader->word0);
      swapByteOrder(mDigitHCHeader->word1);
    }
    if (mVerbose) {
      LOG(info) << mDigitHCHeader->bunchcrossing << " was bunchcrossing and " << mDigitHCHeader->supermodule << " " << mDigitHCHeader->layer;
    }
    if (mHeaderVerbose) {
      printDigitHCHeader(*mDigitHCHeader);
    }
    if (mDigitHCHeader->word0 == 0x0 || mDigitHCHeader->word1 == 0x0) {
      LOG(warn) << "Missing DigitHCHeader, reading digit end marker of zeros??";
      printDigitHCHeader(*mDigitHCHeader);
    }
    mBufferLocation += 2;
    mDataWordsParsed += 2;
    std::advance(mStartParse, 2);
    //move over the DigitHCHeader;
    mState = StateDigitMCMHeader;
  }

  for (auto& word = mStartParse; word < mEndParse; ++word) { // loop over the entire data buffer (a complete link of digits)
    auto looptime = std::chrono::high_resolution_clock::now() - timedigitparsestart;
    //loop over all the words
    if (mDataVerbose || mVerbose) {
      LOG(info) << "parsing word : " << std::hex << *word;
    }
    //check for digit end marker
    if (mByteOrderFix) {
      // byte swap if needed.
      swapByteOrder(*word);
    }
    auto nextword = std::next(word, 1);
    if ((*word) == 0x0 && (*nextword == 0x0)) { // no need to byte swap nextword
      // end of digits marker.
      if (mVerbose || mHeaderVerbose || mDataVerbose) {
        LOG(info) << "Found digits end marker :" << std::hex << *word << "::" << *nextword;
      }
      //state *should* be StateDigitMCMData check that it is
      if (mState == StateDigitMCMData || mState == StateDigitEndMarker || mState == StateDigitHCHeader || mState == StateDigitMCMHeader) {
      } else {

        LOG(error) << "Digit end marker found but state is not StateDigitMCMData(" << StateDigitMCMData << ") or StateDigit but rather " << mState;
      }
      //only thing that can remain is the padding.
      //now read padding words till end.
      //no need to byteswap its uniform.
      mBufferLocation += 2; // 2 words forward.
      mDataWordsParsed += 2;
      mState = StateDigitEndMarker;
    } else {
      if ((*word & 0xf) == 0xc && mState == StateDigitMCMHeader) { //marker for DigitMCMHeader.
        if (mVerbose) {
          LOG(info) << " **** mDigitMCMHeader has value " << std::hex << *word;
        }
        //read the header OR padding of 0xeeee;
        //we actually have an header word.
        mcmadccount = 0;
        mcmdatacount = 0;
        mChannel = 0;
        mDigitMCMHeader = (DigitMCMHeader*)(word);
        if (mVerbose || mHeaderVerbose) {
          LOG(info) << "state mcmheader and word : 0x" << std::hex << *word;
          printDigitMCMHeader(*mDigitMCMHeader);
        }
        if (mDigitHCHeader->major == 0x21){
          //zero suppressed
          //so we have an adcmask next
          std::advance(word, 1);
          if (mByteOrderFix) {
            // byte swap if needed.
            swapByteOrder(*word);
          }
          mDigitMCMADCMask = (DigitMCMADCMask*)(word);
          mADCMask = mDigitMCMADCMask->adcmask;
          //TODO why does the following not hold, its defined as such in the tdp ...  ??? come back
//          if(mDigitMCMADCMask->c!=0x1f || mDigitMCMADCMask->n!=0x3 || mDigitMCMADCMask->j!=0xc){
//            LOG(error) << "ADCMask constant values are not right ; mask: 0x:"<< std::hex << mADCMask << " n c n fullword 0x" << std::hex << mDigitMCMADCMask->n << " 0x" << mDigitMCMADCMask->c << " 0x" << mDigitMCMADCMask->j << " full:0x" << *word;
//        }
          if (mVerbose || mHeaderVerbose) {
            LOG(info) << "**adc mask is " << std::hex << mDigitMCMADCMask->adcmask << " raw form : 0x" << std::hex << mDigitMCMADCMask->word;
          }
          //TODO check for end of loop?
          if (word == mEndParse) {
            LOG(warn) << "we have a problem we have advanced from MCMHeader to the adcmask but are now at the end of the loop";
          }
//output all the adc data for the described adc mask, i.e 10 32 bit words per bit in mask./
          if(mDataVerbose){
            LOG(info) << "RAW ADC DUMP start ";
            std::bitset<21> adcmask(mADCMask);
            int bitsinmask=adcmask.count();
            LOG(info) << " channel account as per bp:"<< std::hex << mADCMask << " has count of :" << std::dec << bitsinmask << "   ADCMask constant values are not right ; mask: 0x:"<< std::hex << mADCMask << " n c n fullword 0x" << std::hex << mDigitMCMADCMask->n << " "<< std::dec << mDigitMCMADCMask->c << " 0x" << std::hex << mDigitMCMADCMask->j << " full:0x" << *word;
            int z;
            for(z=0;z<bitsinmask*10;++z){
              if(z%10==0 && z!=0) LOG(info) <<"gap";
              auto nextadcword=std::next(word,z+1);
              LOG(info)<< "ADCraw: z:"<< z << " " << std::hex << *nextadcword;

            }
            LOG(info) << "next 8 words for clarity";
            int offset=z;
            for(z=offset;z<offset+8;++z){
              auto nextadcword=std::next(word,z+1);
              LOG(info)<< "??ADCraw: z:"<< z << " " << std::hex << *nextadcword;
            }
            LOG(info) << "RAW ADC DUMP end";
          }
        }
          else {
            LOG(info) << "NOT zero suppressed";
          }

          //sanity check of digitheader ??  Still to implement.
          if (mHeaderVerbose) {
            std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator tmpword;

            if (!digitMCMHeaderSanityCheck(mDigitMCMHeader)) {
              LOG(warn) << "Sanity check Failure Digit MCMHeader : " << std::hex << mDigitMCMHeader->word;
            } else {
              LOG(info) << "Sanity check passed for digitmcmheader";
              printDigitMCMHeader(*mDigitMCMHeader);
            }
          }
          mBufferLocation++;
          //new header so digit word count becomes zero
          digitwordcount = 0;
          mState = StateDigitMCMData;
          mMCM = mDigitMCMHeader->mcm;
          mROB = mDigitMCMHeader->rob;
          //cru /2 = supermodule
          //link channel == readoutboard as per guido doc.
          int layer = mDigitHCHeader->layer;
          int stack = mDigitHCHeader->stack;
          int sector = mDigitHCHeader->supermodule;
          mDetector = layer + stack * constants::NLAYER + sector * constants::NLAYER * constants::NSTACK;
          //TODO check that his matches up with the CRU Link info
          //TOOD does it match the feeid which ncodes this information as well.
          //
          mEventCounter = mDigitMCMHeader->eventcount;
          mDataWordsParsed++; // header
          if (mDigitHCHeader->major & 0x20) {
            //zero suppressed digits
            mDataWordsParsed++; // adc mask
          }
          mChannel = 0;
          mADCValues.fill(0);
          digittimebinoffset = 0;
          if (!mReturnVector) {
            //returning the raw "compressed" data stream.
            //build the digit header and add to outgoing buffer;
            uint32_t* header = &(*mData)[mReturnVectorPos];
            //build header.
          }
          // we dont care about the year flag, we are >2007 already.
        } else {
          //LOG(info) << "state not digitmcmheader so checking for other";
          if (*word == o2::trd::constants::CRUPADDING32) {
            if (mVerbose) {
              LOG(info) << "state padding and word : 0x" << std::hex << *word << "  state is:" << mState;
            }
            //another pointer with padding.
            mBufferLocation++;
            mPaddingWordsCounter++;
            mState = StatePadding;
            mDataWordsParsed++;
            mcmdatacount = 0;

            // this is supposed to carry on till the end of the buffer, hence the term padding.
            //TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
          } else { // all we are left with is digitmcmdata words.
            if (mState == StateDigitEndMarker) {

              if (mDataVerbose) {
                LOG(info) << "**DigitEndMarker State : " << std::hex << *word;
                LOG(info) << " digit end marker state ...";
                //we are at the end
                // do nothing.
              }
            } else {
              //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
              //for dpl build a vector and connect it with a triggerrecord.
              if(mDataVerbose){
                LOG(info) << "**Digit word : " << std::hex << *word;
              }
              mDataWordsParsed++;
              mcmdatacount++;
              mDigitMCMData = (DigitMCMData*)word;
              mBufferLocation++;
              mState = StateDigitMCMData;
              digitwordcount++;
              if (mVerbose || mDataVerbose) {
                LOG(info) << "adc values : raw 0x:" << std::hex << *word << std::dec << mDigitMCMData->x << "::" << mDigitMCMData->y << "::" << mDigitMCMData->z << " mask:0x" << std::hex << mADCMask;
                LOG(info) << "digittimebinoffset = " << digittimebinoffset;
              }
              mADCValues[digittimebinoffset++] = mDigitMCMData->x;
              mADCValues[digittimebinoffset++] = mDigitMCMData->y;
              mADCValues[digittimebinoffset++] = mDigitMCMData->z;

              if (digittimebinoffset > constants::TIMEBINS) {
                LOG(fatal) << "too many timebins to insert into mADCValues digittimebinoffset:" << digittimebinoffset;
              }
              if (mVerbose || mDataVerbose) {
                LOG(info) << "digit word count is : " << digitwordcount << " digittimebinoffset = " << digittimebinoffset;
              }
              if (digitwordcount == constants::TIMEBINS / 3) {
                //sanity check, next word shouldbe either a. end of digit marker, digitMCMHeader,or padding.
                mcmadccount++;
                //write out adc value to vector
                //zero digittimebinoffset
                if (mDigitHCHeader->major & 0x20) {
                  //zero suppressed, so channel must be extracted from next available bit in adcmask
                  mChannel = nextmcmadc(mADCMask, mChannel);
                  if(mChannel==21){
                    LOG(warn)<< "ADCMask is zero but we seem to have a digit";
                  }
                  if (mDataVerbose) {
                    LOG(info) << "after mask check adcmask: 0x" << std::hex << mADCMask << " and channel : " << std::dec << mChannel;
                    LOG(info) << "the above is the preceding digit above us not the one below us ";
                  }
                  if (mChannel == 22) {
                    LOG(error) << "invalid bitpattern for this mcm 0x"<< std::hex << mADCMask;
                    LOG(info) << "EEE " << mDetector << ":" << mROB << ":" << mMCM << ":" << mChannel << " :" ;
                    for(auto adc : mADCValues ){
                      std::cout << adc << ":" ;
                    }
                    std::cout << std::endl;
                  }
                  if (mADCMask == 0) {
                    LOG(info) << "mADCMask is now zero ";
                    //no more adc for zero suppression.
                    //now we should either have another MCMHeader, or End marker
                    if (*word != 0 && *(std::next(word)) != 0) { // end marker is a sequence of 32 bit 2 zeros.
                             LOG(info)<< "Mask zero so changing state to StateDigitMCMHeader to read an MCMHeader on the next pass next 2 words are not digit end marker of zero";
                      mState = StateDigitMCMHeader;
                    } else {
                            LOG(info)<< "Mask zero so changing state to StateDigitEndHeader as next 2 words are digit end marker of zero 0x"<< *word << " " << *(std::next(word));
                      mState = StateDigitEndMarker;
                    }
                  }
                }
                mDigits.emplace_back(mDetector, mROB, mMCM, mChannel, mADCValues); // outgoing parsed digits
                if(mDataVerbose){
                  //    CompressedDigit t = mDigits.back();
                  //now fill in the adc values --- here because in commented code above if all 3 increments were there then it froze
                  uint32_t adcsum = 0;
                  for (auto adc : mADCValues) {
                    adcsum += adc;
                  }
                  LOG(info) << "DDDD #" << mDigitsFound << " det:rob:mcm:channel:adcsum ... "<< mDetector << ":" << mROB << ":" << mMCM << ":" << mChannel << ":" << adcsum << ":" << mADCValues[0] << ":" << mADCValues[1] << ":" << mADCValues[2] << "::" << mADCValues[27] << ":" << mADCValues[28] << ":" << mADCValues[29] ;
                }
                mDigitsFound++;
                digittimebinoffset = 0;
                digitwordcount = 0; // end of the digit.
                if (mDigitHCHeader->major == 5) {
                  mChannel++; // we count channels as all 21 channels are present, no way to check this.
                }
              }// digitwordcount ==z timebins/3
            }//else digitendmarker if statement
          }// else of if crupadding word
        }// else of if digitMCMheader
    }// else of if digitendmarker
    //accounting
    // mCurrentLinkDataPosition256++;
    // mCurrentHalfCRUDataPosition256++;
    // mTotalHalfCRUDataLength++;
    //end of data so
  }// for loop over word
  if (mVerbose) {
    LOG(info) << "***** parsing loop finished for this link";
  }
  if (!(mState == StateDigitMCMHeader || mState == StatePadding || mState == StateDigitEndMarker)) {
    LOG(warn) << "Exiting parsing but the state is wrong ... mState= " << mState;
  }
  return mDataWordsParsed;
}

} // namespace o2::trd
