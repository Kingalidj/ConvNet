#pragma once

#include "Tensor.h"

#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

namespace Compass
{
	enum class LayerType
	{
		CONVOLUTION = 0,
		POOL,
		ACTIVATION,
		RELU,
		FULLY_CONNECTED,
		NONE
	};

	struct Gradient
	{
		float Grad = 0;
		float OldGrad = 0;

		Gradient() = default;
	};

	struct Range
	{
		int MinX, MinY, MinZ;
		int MaxX, MaxY, MaxZ;
	};

	static float UpdateWeight(float w, Gradient& grad, float multp = 1)
	{
		float m = (grad.Grad + grad.OldGrad * MOMENTUM);
		w -= LEARNING_RATE * m * multp +
			LEARNING_RATE * WEIGHT_DECAY * w;
		return w;
	}

	static void UpdateGradient(Gradient& grad)
	{
		grad.OldGrad = (grad.Grad + grad.OldGrad * MOMENTUM);
	}


	class Layer
	{
	protected:
		LayerType m_Type = LayerType::NONE;

		std::shared_ptr<Tensor<float>> m_Gradient;
		std::shared_ptr<Tensor<float>> m_Input;
		std::shared_ptr<Tensor<float>> m_Output;

	public:
		Layer() = default;
		~Layer() = default;

		Layer(LayerType type, TensorSize inputSize, TensorSize OutputSize)
			: m_Gradient(std::make_shared<Tensor<float>>(inputSize)), m_Input(std::make_shared<Tensor<float>>(inputSize)), m_Output(std::make_shared<Tensor<float>>(OutputSize))
		{
		}

		virtual void Activate(std::shared_ptr<Tensor<float>> tensor) = 0;
		virtual void UpdateWeights() = 0;
		virtual void ComputeGradient(std::shared_ptr<Tensor<float>> nextLayer) = 0;

		const std::shared_ptr<Tensor<float>>& GetInput() { return m_Input; }
		const std::shared_ptr<Tensor<float>>& GetOutput() { return m_Output; }
		const std::shared_ptr<Tensor<float>>& GetGradient() { return m_Gradient; }
		const LayerType GetLayerType() const { return m_Type; }
	};
}