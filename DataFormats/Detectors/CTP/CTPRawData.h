#ifndef ALICEO2_CTPRAWDATA_H	#ifndef ALICEO2_CTPRAWDATA_H
#define ALICEO2_CTPRAWDATA_H	#define ALICEO2_CTPRAWDATA_H
#include InteractionRecord.h	#include InteractionRecord.h
namespace  o2	namespace o2
{	{
    namespace ctp	namespace ctp
    {	{
        struct CTPCRUData	struct CTPCRUData {
        {	  static constexpr uint8_t NumberOfClasses = 64;
            static constexpr uint8_t NumberOfClasses=64;	  static constexpr uint8_t NumberOfLMinputs = 16;
            static constexpr uint8_t NumberOfLMinputs=16;	  static constexpr uint8_t NumberOfL0inputs = 30;
            static constexpr uint8_t NumberOfL0inputs=30;	  static constexpr uint8_t NumberOfL1inputs = 18;
            static constexpr uint8_t NumberOfL1inputs=18;	  InteractionRecord ir;
            InteractionRecord ir;	  uint32_t InputsMask = 0;
            uint32_t InputsMask=0;
