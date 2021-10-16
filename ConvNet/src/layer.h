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

		std::shared_ptr<Tensor<T>> m_Gradiens;
		std::shared_ptr<Tensor<T>> m_Input;
		std::shared_ptr<Tensor<T>> m_Output;

	public:
		Layer() = default;
		~Layer() = default;

		Layer(LayerType type, TensorSize inputSize, TensorSize OutputSize)
			: m_Gradiens(std::make_shared<Tensor<T>>(inputSize)), m_Input(std::make_shared<Tensor<T>>(inputSize)), m_Output(std::make_shared<Tensor<T>>(OutputSize))
		{
		}
	};
}