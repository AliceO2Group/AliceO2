#ifndef STANDALONESETTINGS_H
#define STANDALONESETTINGS_H

struct hltca_event_dump_settings
{
	void setDefaults()
	{
		solenoidBz = -5.00668;
		constBz = false;
		homemadeEvents = false;
	}
	
	float solenoidBz;
	bool constBz;
	bool homemadeEvents;
} __attribute__((packed));

#endif
