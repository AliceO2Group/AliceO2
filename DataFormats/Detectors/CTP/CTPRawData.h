#ifndef ALICEO2_CTPRAWDATA_H
#define ALICEO2_CTPRAWDATA_H
#include InteractionRecord.h
namespace  o2
{
    namespace ctp
    {
        struct CTPCRUData
        {
            static constexpr uint8_t NumberOfClasses=64;
            static constexpr uint8_t NumberOfLMinputs=16;
            static constexpr uint8_t NumberOfL0inputs=30;
            static constexpr uint8_t NumberOfL1inputs=18;
            InteractionRecord ir;
            uint32_t InputsMask=0;
            uint64_t TriggerClassMask=0;

            CTPRawData() = default;
        }
    }

}

