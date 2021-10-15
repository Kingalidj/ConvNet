#pragma once

#include "common.h"
#include "Tensor.h"

namespace Compass
{
	template <typename T>
	class Layer
	{
	protected:
		LayerType m_Type = LayerType::NONE;

		Tensor<T> m_Gradiens;
		Tensor<T> m_Input;
		Tensor<T> m_Output;

		Layer(LayerType type)
			: m_Type(type) {}

	public:
		Layer() = default;
	};
}