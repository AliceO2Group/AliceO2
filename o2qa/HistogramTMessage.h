#pragma once

#include <TMessage.h>
#include <string>

using namespace std;

class HistogramTMessage : public TMessage
{
public:
    HistogramTMessage(void* buf, Int_t len)
        : TMessage(buf, len)
    {
        ResetBit(kIsOwner);
    }
};

