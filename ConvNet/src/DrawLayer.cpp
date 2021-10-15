#include "DrawLayer.h"

#include "Tensor.h"
#include "ConvolutionLayer.h"
#include "Random.h"

#include <ImGui/imgui.h>

using namespace Atlas;

DrawLayer::DrawLayer()
{
	Compass::Random::Init();
}

void DrawLayer::OnAttach()
{
	Compass::Tensor<float> t1 = Compass::Tensor<float>(10, 10, 10);
	Compass::RandomizeTensor<float>(t1, -1, 1);

	tFB = { t1 };

	Compass::ConvolutionLayer<float> layer = Compass::ConvolutionLayer<float>({ 1, 1, 1 });
}

void DrawLayer::OnDetach()
{
}

void DrawLayer::OnUpdate(Timestep ts)
{
}


void DrawLayer::OnImGuiRender()
{
	ImGui::Begin("ConvNet Viewer");
	ImGui::Image((void*)(std::size_t)tFB.GetRendererID(), ImVec2(300, 300));
	ImGui::End();

}

void DrawLayer::OnEvent(Event& e)
{
}
