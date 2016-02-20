#pragma once

class TObject;

class Producer
{
	public:
		virtual TObject* produceData() const = 0;	
};
