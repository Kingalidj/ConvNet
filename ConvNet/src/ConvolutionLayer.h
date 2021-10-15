#pragma once

#include "layer.h"

#include <vector>

namespace Compass
{

	template<typename T>
	class ConvolutionLayer : public Layer<T>
	{
	private:
		std::vector<Tensor<T>> m_Filters;

		uint16_t m_Stride, m_ExtendFilter;

	public:
		ConvolutionLayer(TensorSize inputSize)
			: Layer(LayerType::CONV)
		{
		}

	};

}
