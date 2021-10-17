#pragma once

#include <Atlas.h>
#include "TensorFrameBuffer.h"
#include "ConvolutionLayer.h"

class DrawLayer : public Atlas::Layer
{
private:
	//Compass::TensorFrameBuffer tFB;
	Compass::ConvolutionLayer<float> m_Layer;
	std::vector<Compass::TensorFrameBuffer<float>> m_Textures;
	Compass::Tensor<float> m_Input = { 28, 28, 1 };
	Compass::TensorFrameBuffer<float> m_InputTexture;

public:

	DrawLayer();
	virtual void OnAttach() override;
	virtual void OnDetach() override;

	void OnUpdate(Atlas::Timestep ts) override;
	virtual void OnImGuiRender() override;
	void OnEvent(Atlas::Event& e) override;
};

