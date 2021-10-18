#pragma once

#include "Tensor.h"

#include <Atlas/Renderer/Texture.h> 

namespace Compass
{

	class TensorFrameBuffer
	{
		uint32_t m_Width = 0, m_Height = 0;
		uint32_t m_Depth;

	private:
		std::vector<std::shared_ptr<Atlas::Texture2D>> m_TensorTexture;
		std::shared_ptr<Tensor<float>> m_Tensor;

	public:

		TensorFrameBuffer() = default;

		TensorFrameBuffer(std::shared_ptr<Tensor<float>> tensor)
			: m_Width(tensor->GetWidth()), m_Height(tensor->GetHeight()), m_Depth(tensor->GetDepth()), m_Tensor(tensor)
		{
			unsigned char* data = new unsigned char[m_Height * m_Width * 4];

			for (uint32_t k = 0; k < m_Depth; k++)
			{
				for (uint32_t i = 0; i < m_Width; i++)
				{
					for (uint32_t j = 0; j < m_Height; j++)
					{
						unsigned char red = 0;
						unsigned char blue = 0;
						float value = tensor->GetData(i, j, k);
						if (value < 0) red = static_cast<unsigned char>(-255.0 * value);
						if (tensor->GetData(i, j, 0) > 0) blue = static_cast<unsigned char>(255.0 * tensor->GetData(i, j, 0));

						data[(j * m_Width * 4) + i * 4] = red;
						data[(j * m_Width * 4) + i * 4 + 1] = 0;
						data[(j * m_Width * 4) + i * 4 + 2] = blue;
						data[(j * m_Width * 4) + i * 4 + 3] = 255;
					}
				}
				Atlas::Ref<Atlas::Texture2D> texture = Atlas::Texture2D::Create(m_Width, m_Height);
				texture->SetData(data, m_Height * m_Width * 4 * sizeof(unsigned char));
				m_TensorTexture.push_back(texture);
			}


			delete[] data;
		}

		uint32_t GetRendererID(uint32_t indx = 0) { return m_TensorTexture[indx]->GetRendererID(); }

		void Update()
		{
			unsigned char* data = new unsigned char[m_Height * m_Width * 4];

			for (uint32_t k = 0; k < m_Depth; k++)
			{
				for (uint32_t i = 0; i < m_Width; i++)
				{
					for (uint32_t j = 0; j < m_Height; j++)
					{
						unsigned char red = 0;
						unsigned char blue = 0;
						float value = m_Tensor->GetData(i, j, 0);
						if (value < 0) red = static_cast<unsigned char>(-255.0 * value);
						if (value > 0) blue = static_cast<unsigned char>(255.0 * value);

						data[(i * m_Width * 4) + j * 4] = red;
						data[(i * m_Width * 4) + j * 4 + 1] = 0;
						data[(i * m_Width * 4) + j * 4 + 2] = blue;
						data[(i * m_Width * 4) + j * 4 + 3] = 255;
					}
				}
				m_TensorTexture[k]->SetData(data, m_Height * m_Width * 4 * sizeof(unsigned char));
			}

		}
	};
}
