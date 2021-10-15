#pragma once

#include <Atlas.h>
#include "TensorFrameBuffer.h"

class DrawLayer : public Atlas::Layer
{
private:
	Compass::TensorFrameBuffer tFB;

public:

	DrawLayer();
	virtual void OnAttach() override;
	virtual void OnDetach() override;

	void OnUpdate(Atlas::Timestep ts) override;
	virtual void OnImGuiRender() override;
	void OnEvent(Atlas::Event& e) override;
};

