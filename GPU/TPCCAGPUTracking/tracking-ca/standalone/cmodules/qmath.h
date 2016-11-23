static inline bool qIsFinite(double val)
{
	const unsigned long long int ival = *((unsigned long long int*) &val);
	return (!((ival & 0x7FF0000000000000) == 0x7FF0000000000000));
}