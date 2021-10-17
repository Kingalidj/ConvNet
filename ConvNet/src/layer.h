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

	public:
		Layer() = default;
		~Layer() = default;

		Layer(LayerType type, TensorSize inputSize, TensorSize OutputSize)
			: m_Gradiens(inputSize), m_Input(inputSize), m_Output(OutputSize)
		{
		}
	};
}