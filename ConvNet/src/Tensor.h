#pragma once

#include "Random.h"

#include <cstdint>
#include <functional>
#include <Atlas/Core/Core.h>

namespace Compass
{
	struct TensorSize
	{
		uint32_t Width, Height, Depth;
	};

	template<typename T>
	class Tensor
	{
	private:
		T* m_Values = nullptr;

		uint32_t m_Width = 0, m_Height = 0, m_Depth = 0;

	public:
		Tensor() {}

		Tensor(std::uint32_t col, std::uint32_t row, std::uint32_t depth)
			: m_Width(col), m_Height(row), m_Depth(depth)
		{
			m_Values = new T[(std::size_t) m_Width * m_Height * m_Depth];
		}

		Tensor(TensorSize size)
			: m_Width(size.Width), m_Height(size.Height), m_Depth(size.Depth)
		{
			m_Values = new T[(std::size_t) m_Width * m_Height * m_Depth];
		}

		Tensor(std::uint32_t col, std::uint32_t row, std::uint32_t depth, T value)
			: m_Width(col), m_Height(row), m_Depth(depth)
		{
			m_Values = new T[(std::size_t) m_Width * m_Height * m_Depth];

			for (uint32_t i = 0; i < m_Depth * m_Width * m_Height; i++)
			{
				m_Values[i] = value;
			}
		}

		~Tensor()
		{
			if (m_Values != NULL)
			{
				delete[] m_Values;
			}
		}

		Tensor(const Tensor& other)
			: m_Width(other.m_Width), m_Height(other.m_Height), m_Depth(other.m_Depth), m_Values(new T[(std::size_t) other.m_Width * other.m_Height * other.m_Depth])
		{
			memcpy(m_Values, other.m_Values, (std::size_t) m_Width * m_Height * m_Depth);
		}

		T& operator()(std::uint32_t x, std::uint32_t y, std::uint32_t z) const
		{
			return GetData(x, y, z);
		}

		Tensor& operator+=(const Tensor& other)
		{
			for (uint32_t i = 0; i < m_Depth * m_Width * m_Height; i++)
			{
				m_Values[i] += other.m_Values[i];
			}

			return *this;
		}

		Tensor& operator-=(const Tensor& other)
		{
			for (uint32_t i = 0; i < m_Depth * m_Width * m_Height; i++)
			{
				m_Values[i] -= other.m_Values[i];
			}

			return *this;
		}

		void MapFunction(const std::function<void(T&)>& function)
		{
			for (uint32_t i = 0; i < m_Depth * m_Width * m_Height; i++)
			{
				function(m_Values[i]);
			}
		}

		void Fill(T value)
		{
			for (uint32_t i = 0; i < m_Depth * m_Width * m_Height; i++)
			{
				m_Values[i] = value;
			}
		}

		T& GetData(std::uint32_t x, std::uint32_t y, std::uint32_t z) const
		{
			ATL_ASSERT(x >= 0 && y >= 0 && z >= 0, "Index out of bounds");
			ATL_ASSERT(x < m_Width && y < m_Height && z < m_Depth, "Index out of bounds");

			return m_Values[z * (m_Width * m_Height) + y * m_Width + x];
		}

		T* GetValuePointer() const { return m_Values; }
		const uint32_t GetHeight() const { return m_Height; }
		const uint32_t GetWidth() const { return m_Width; }
		const uint32_t GetDepth() const { return m_Depth; }
	};

	template<typename T>
	Tensor<T> operator+(Tensor<T> t1, const Tensor<T>& t2)
	{
		t1 += t2;
		return t1;
	}

	template<typename T>
	Tensor<T> operator-(Tensor<T> t1, const Tensor<T>& t2)
	{
		t1 -= t2;
		return t1;
	}

	template<typename T>
	static void RandomizeTensor(Tensor<T>& tensor, T min, T max)
	{
		ATL_CORE_ERROR("Randomization for type {0} is not supported", typeid(T).name());
		ATL_ASSERT(false, "");
	}

	template<>
	static void RandomizeTensor<float>(Tensor<float>& tensor, float min, float max)
	{
		tensor.MapFunction([&](float& a) { a = static_cast<float>(Random::RandRangeDouble(min, max)); });
	}

	template<>
	static void RandomizeTensor<double>(Tensor<double>& tensor, double min, double max)
	{
		tensor.MapFunction([&](double& a) { a = Random::RandRangeDouble(min, max); });
	}

	template<>
	static void RandomizeTensor<int>(Tensor<int>& tensor, int min, int max)
	{
		tensor.MapFunction([&](int& a) { a = Random::RandRangeInt(min, max); });
	}
}

