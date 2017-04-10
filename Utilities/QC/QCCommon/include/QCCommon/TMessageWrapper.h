
#pragma once

#include <TMessage.h>

class TMessageWrapper : public TMessage
{
public:
    TMessageWrapper(void* buf, Int_t len) : TMessage(buf, len)
    {
        ResetBit(kIsOwner);
    }

    ~TMessageWrapper() override = default;
};
