#ifndef Q_BITFIELD_H
#define Q_BITFIELD_H

template <class T, class S> class bitfield
{
public:
	bitfield(T v) : bits((S) v) {}
	bitfield(S v = 0) : bits(v) {}
	bitfield(const bitfield&) = default;
	bitfield& operator=(const bitfield&) = default;
	bitfield operator|(const bitfield v) const {return bits | v.bits;}
	bitfield& operator|=(const bitfield v) {bits |= v.bits; return *this;}
	bitfield operator&(const bitfield v) const {return bits & v.bits;}
	bitfield operator&(const T v) const {return bits & (S) v;}
	bitfield& operator&=(const bitfield v) {bits &= v.bits; return *this;}
	bitfield operator~() const {return ~bits;}
	bool operator==(const bitfield v) {return bits == v.bits;}
	bool operator==(const T v) {return bits == (S) v;}
	bool operator!=(const bitfield v) {return bits != v.bits;}
	bool operator!=(const T v) {return bits != (S) v;}
	bitfield& setBits(const bitfield v, bool w) {if (w) bits |= v; else bits &= ~v; return *this;}
	void set(S v) {bits = v;}
	operator bool() const {return bits;}
	explicit operator S() const {return bits;}
	bool isSet(const bitfield& v) const {return *this & v;}
	
private:
	S bits;
};

#endif
