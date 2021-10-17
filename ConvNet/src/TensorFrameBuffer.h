#pragma once

#include "Tensor.h"

#include <Atlas/Renderer/Texture.h> 

namespace Compass
{

	template<typename T>
	class TensorFrameBuffer
	{
		uint32_t m_Width = 0, m_Height = 0;

	private:
		std::shared_ptr<Atlas::Texture2D> m_TensorTexture;
		Tensor<T>* m_Tensor;

	public:

		TensorFrameBuffer() = default;

		TensorFrameBuffer(Tensor<T>& tensor)
			: m_Width(tensor.GetWidth()), m_Height(tensor.GetHeight()), m_Tensor(&tensor)
		{
			unsigned char* data = new unsigned char[m_Height * m_Width * 4];

			for (uint32_t i = 0; i < m_Width; i++)
			{
				for (uint32_t j = 0; j < m_Height; j++)
				{
					unsigned char red = 0;
					unsigned char blue = 0;
					if (tensor.GetData(i, j, 0) < 0) red = static_cast<unsigned char>(-255.0 * tensor.GetData(i, j, 0));
					if (tensor.GetData(i, j, 0) > 0) blue = static_cast<unsigned char>(255.0 * tensor.GetData(i, j, 0));

					data[(i * m_Width * 4) + j * 4] = red;
					data[(i * m_Width * 4) + j * 4 + 1] = 0;
					data[(i * m_Width * 4) + j * 4 + 2] = blue;
					data[(i * m_Width * 4) + j * 4 + 3] = 255;
				}
			}

			m_TensorTexture = Atlas::Texture2D::Create(m_Width, m_Height);
			m_TensorTexture->SetData(data, m_Height * m_Width * 4 * sizeof(unsigned char));

			delete[] data;
		}

		uint32_t GetRendererID() { return m_TensorTexture->GetRendererID(); }
		Tensor<T>* GetValuePointer() { return m_Tensor; }

		void Update()
		{
			if (m_Tensor)
			{
				unsigned char* data = new unsigned char[m_Height * m_Width * 4];

				for (uint32_t i = 0; i < m_Width; i++)
				{
					for (uint32_t j = 0; j < m_Height; j++)
					{
						unsigned char red = 0;
						unsigned char blue = 0;
						T value = m_Tensor->GetData(i, j, 0);
						if (value < 0) red = static_cast<unsigned char>(-255.0 * value);
						if (value > 0) blue = static_cast<unsigned char>(255.0 * value);

						data[(i * m_Width * 4) + j * 4] = red;
						data[(i * m_Width * 4) + j * 4 + 1] = 0;
						data[(i * m_Width * 4) + j * 4 + 2] = blue;
						data[(i * m_Width * 4) + j * 4 + 3] = 255;
					}
				}

				m_TensorTexture->SetData(data, m_Height * m_Width * 4 * sizeof(unsigned char));

			}
		}
	};
}
