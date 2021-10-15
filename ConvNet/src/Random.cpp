#include "Random.h"

#include <time.h>
#include <random>

namespace Compass
{
	static std::mt19937 m_RNG;
	static std::uniform_int_distribution<uint32_t> m_UIntDist;
	static std::uniform_int_distribution<int> m_IntDist;
	static std::uniform_real_distribution<double> m_DoubleDist;

	void Random::Init()
	{
		m_RNG.seed(static_cast<uint32_t>(time(NULL)));
	}

	uint32_t Random::RandUInt()
	{
		return m_UIntDist(m_RNG);
	}

	int Random::RandInt()
	{
		return m_IntDist(m_RNG);
	}

	double Random::RandDouble()
	{
		return m_DoubleDist(m_RNG);
	}

	double Random::RandRangeDouble(double min, double max)
	{
		return min + (RandDouble() * (max - min));
	}

	int Random::RandRangeInt(int min, int max)
	{
		return min + (int)(RandDouble() * ((max - min) + 1));
	}

}
