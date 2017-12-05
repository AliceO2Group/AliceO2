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
	
	//New members should always go to the end, the packed attribute and the reading will make sure new members are initialized to defaults when reading old files
	float solenoidBz;
	bool constBz;
	bool homemadeEvents;
} __attribute__((packed));

#endif
