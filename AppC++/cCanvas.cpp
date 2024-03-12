#include "cCanvas.h"
#include <string>
#include <fstream>
#include "Histogram.h"
#include <Eigen/Dense>
#include "computations.h"





wxBEGIN_EVENT_TABLE(cCanvas, wxHVScrolledWindow)
EVT_PAINT(cCanvas::OnPaint)
wxEND_EVENT_TABLE()
// agregar logica para un nuevo constructor dado un histograma y que en la funcion de dibujar compruebe si sse va a dibujar un histograma 

// mandarle la imagen de el histograma y solamente dibujar mas 
cCanvas::cCanvas(wxWindow* parent, wxString filename) : wxHVScrolledWindow(parent, wxID_ANY)  // ----- FINISHED
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	Bind(wxEVT_RIGHT_UP, &cCanvas::OnMouseClick, this);
	Bind(wxEVT_LEFT_UP, &cCanvas::OnMouseClickKill, this);
	SetRowColumnCount(40, 40); // see the changes if change that values
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	this->fileName = filename;
	this->img_load = true;
}

cCanvas::cCanvas(wxWindow* parent, unsigned char* data, int w, int h) : wxHVScrolledWindow(parent, wxID_ANY) // constuctor given a new image - FINISHED
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	Bind(wxEVT_RIGHT_UP, &cCanvas::OnMouseClick, this);
	Bind(wxEVT_LEFT_UP, &cCanvas::OnMouseClickKill, this);
	SetRowColumnCount(40, 40); // see the changes if change that values
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	this->m_myImage = data;
	this->m_imageWidth = w;
	this->m_imageHeight = h;
	this->m_imageRGB = new wxImage(this->m_imageWidth, this->m_imageHeight, this->m_myImage, true);
	this->img_load = true;
}

cCanvas::cCanvas(wxWindow* parent, Histogram* h) // constructor to create a histogram given a image  // ----- FINISHED
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	Bind(wxEVT_RIGHT_UP, &cCanvas::OnMouseClick, this);
	Bind(wxEVT_LEFT_UP, &cCanvas::OnMouseClickKill, this);
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	this->hist = h;
	this->m_imageRGB = nullptr;
	this->m_myImage = nullptr;
	this->img_load = false;
	this->m_imageHeight = 0;
	this->m_imageWidth = 0;

}

int cCanvas::getWidth()  // ----- FINISHED
{
	return this->m_imageWidth;
}

int cCanvas::getHeight()  // ----- FINISHED
{
	return this->m_imageHeight;
}

cCanvas::~cCanvas()  // ----- FINISHED
{
	delete this->hist;
	this->matrixClasses.clear();
	delete this->m_imageRGB;
	this->m_imageRGB = nullptr;
	free(this->m_myImage);
	this->m_myImage = nullptr;
}

void cCanvas::setPixelSize(int p)  // ----- FINISHED
{
	m_nPixelSize = p; // se the changes of that variable
	wxVarHScrollHelper::RefreshAll();
	wxVarVScrollHelper::RefreshAll();
	Refresh();
}

void cCanvas::OnPaint(wxPaintEvent& event) // pre painting, the paint handle event
{
	//wxLogMessage("Entramos aqui");
	wxBufferedPaintDC dc(this);
	this->PrepareDC(dc);
	this->OnDraw(dc);
}

void cCanvas::OnDraw(wxDC& dc) // Arregla esta problematica para dibujar la imagen 
{

	dc.Clear();
	wxBrush brush = dc.GetBrush();
	wxPen pen = dc.GetPen();


	dc.SetBrush(brush);
	wxImage* tempImage;

	if (this->hist == nullptr)
	{
		tempImage = new wxImage(m_imageWidth, m_imageHeight, m_myImage, true); // lend my image buffer...
		m_imageBitmap = wxBitmap(*tempImage, -1); // ...to get the corresponding bitmap
		delete(tempImage);		// buffer not needed any more
		dc.DrawBitmap(this->m_imageBitmap, 0, 0);
		for (int i = 0; i < this->rectangles.size(); i += 2)
		{
			if (i + 1 < this->rectangles.size())
			{
				// Calcula el ancho y la altura
				int width = this->rectangles[i + 1].x - this->rectangles[i].x;
				int height = this->rectangles[i + 1].y - this->rectangles[i].y;

				// Si width o height son negativos, ajusta el punto de origen
				int x = width < 0 ? this->rectangles[i + 1].x : this->rectangles[i].x;
				int y = height < 0 ? this->rectangles[i + 1].y : this->rectangles[i].y;

				//  width y height sean positivos para DrawRectangle
				width = abs(width);
				height = abs(height);

				// Establecer el color y el grosor del borde del rectángulo
				dc.SetPen(wxPen(wxColour(0, 0, 0), 2)); // Negro y 2 píxeles de grosor
				int numClass = i / 2;
				// Establecer el pincel como transparente para no rellenar el rectángulo
				wxString text = wxString::Format(wxT("clase : %d"), numClass);
				dc.SetTextForeground(*wxGREEN);
				dc.DrawText(text, x, y - 20);
				dc.SetBrush(*wxTRANSPARENT_BRUSH);
				dc.DrawRectangle(x, y, width, height);
			}
		}
	}
	else
	{
		dc.DrawRectangle(10, 10, 100, 50);
		wxMessageBox(wxT("Entramos al hist"));
	}

}
unsigned char* cCanvas::ToGray()  // ----- FINISHED
{
	// generar un buffer que guarda los nuevos valores, y retornar eso, para asi crear una nueva ventana

	if (this->getformat() == (wxString)"RGB" || this->getformat() == (wxString)"RGBA")
	{
		//wxMessageBox(wxT("RGB to gray"));
		unsigned char* temp = (unsigned char*)malloc(m_imageWidth * m_imageHeight * 3);
		int long pixels = m_imageHeight * m_imageWidth;
		for (int i = 0; i < pixels; ++i)
		{
			unsigned char red = this->m_myImage[i * 3];
			unsigned char green = this->m_myImage[i * 3 + 1];
			unsigned char blue = this->m_myImage[i * 3 + 2];
			unsigned char gray = (unsigned char)(0.299 * red + 0.587 * green + 0.114 * blue);
			temp[i * 3] = gray;
			temp[i * 3 + 1] = gray;
			temp[i * 3 + 2] = gray;
		}
		return temp;

	}
	else if (this->getformat() == (wxString)"GRAY")
	{
		wxMessageBox(wxT("Already Gray"));
		return nullptr;
	}
	else
	{
		wxMessageBox(wxT("Not implemented..."));
		return nullptr;
	}

}
wxString cCanvas::getformat()   // ----- FINISHED
{
	for (int i = 0; i < this->m_imageHeight; i++)
	{
		for (int j = 0; j < this->m_imageWidth; j++)
		{
			int offset = (i * this->m_imageWidth + j) * 3; // 3 canales (RGB)
			unsigned char red = this->m_myImage[offset] - 0;
			unsigned char green = this->m_myImage[offset + 1] - 0;
			unsigned char blue = this->m_myImage[offset + 2] - 0;
			//ofs << "[" << red + ","<< green << "," << blue << "]";
			if (red != green || green != blue)
			{
				if (this->m_imageRGB->HasAlpha())
				{
					return (wxString)"RGBA";
				}
				else
				{
					return (wxString)"RGB";
				}
			}
		}
	}
	return (wxString)"GRAY";


}
void cCanvas::saveImage(wxString filename)  // ----- FINISHED
{
	bool b;

	wxImage* tempImage = new wxImage(this->m_imageWidth, this->m_imageHeight, this->m_myImage, true); // lend my image buffer...
	b = tempImage->SaveFile(filename);
	delete(tempImage);		// buffer not needed any more

	if (!b)
		wxMessageBox(wxT("A problem occured during saving"));
}
void cCanvas::LoadImage()  // ----- FINISHED
{
	if (m_myImage)
		free(m_myImage);
	if (m_imageRGB)
		delete m_imageRGB;

	// open image dialog box
	m_imageRGB = new wxImage(this->fileName, wxBITMAP_TYPE_ANY, -1); // ANY => can load many image formats
	m_imageBitmap = wxBitmap(*m_imageRGB, -1); // ...to get the corresponding bitmap

	m_imageWidth = m_imageRGB->GetWidth();
	m_imageHeight = m_imageRGB->GetHeight();

	m_myImage = (unsigned char*)malloc(m_imageWidth * m_imageHeight * 3);
	memcpy(m_myImage, m_imageRGB->GetData(), m_imageWidth * m_imageHeight * 3);
	if (this->m_myImage == nullptr)
	{
		wxString message = wxT("Image carge failed") + wxString::Format(wxT("%d x %d"), m_imageWidth, m_imageHeight);
		wxMessageBox(message);
		return;
	}
	// update display
	//this->getformat();
	// agregar aqui para probar un filtro en la imagen para probar como se obtienen los valores 
	for (int x = 0; x < m_imageWidth; ++x) {
		for (int y = 0; y < m_imageHeight; ++y) {
			// Cast explícito de unsigned char a int
			int r = static_cast<double>(m_myImage[(y * m_imageWidth + x) * 3 + 0]);
			int g = static_cast<double>(m_myImage[(y * m_imageWidth + x) * 3 + 1]);
			int b = static_cast<double>(m_myImage[(y * m_imageWidth + x) * 3 + 2]);

			// Guarda los valores RGB en el diccionario con la clave siendo las coordenadas (x, y)
			this->pixelColors[std::make_pair(x, y)] = std::make_tuple(r, g, b);
		}
	}
	this->SetSize(m_imageHeight, m_imageWidth);


	Refresh(false);
}
wxCoord cCanvas::OnGetRowHeight(size_t row) const  // ----- FINISHED
{
	return wxCoord(m_nPixelSize);
}

wxCoord cCanvas::OnGetColumnWidth(size_t col) const // ----- FINISHED
{
	return wxCoord(m_nPixelSize);
}
Histogram* cCanvas::getHist() // ----- FINISHED
{
	long int max_frec = 0;
	long int frec_r[256] = { 0 };
	long int frec_g[256] = { 0 };
	long int frec_b[256] = { 0 };
	if (this->getformat() == (wxString)"RGB" || this->getformat() == (wxString)"RGBA")
	{

		wxImageHistogram hist;
		unsigned long colors = this->m_imageRGB->ComputeHistogram(hist);
		for (auto iter = hist.begin(); iter != hist.end(); ++iter) {
			unsigned long color = iter->first;
			unsigned long value = iter->second.value;

			// Extracting RGB values from the color
			unsigned char red = (unsigned char)(color >> 16);
			frec_r[red] += 1;
			if (frec_r[red] > max_frec) max_frec = frec_r[red];
			unsigned char green = (unsigned char)(color >> 8);
			frec_g[green] += 1;
			if (frec_g[green] > max_frec) max_frec = frec_g[green];
			unsigned char blue = (unsigned char)color;
			frec_b[blue] += 1;
			if (frec_b[blue] > max_frec) max_frec = frec_b[blue];
		}
		for (int i = 0; i < 256; i++)
		{
			wxLogMessage("Color %i (%li, %li, %li) and the maxf is  %li", i, frec_r[i], frec_g[i], frec_b[i], max_frec);
		}
	}
	else if (this->getformat() == (wxString)"GRAY")
	{
		wxImageHistogram histogram;
		this->m_imageRGB->ComputeHistogram(histogram);
		for (auto iter = histogram.begin(); iter != histogram.end(); ++iter)
		{
			unsigned long color = iter->first;
			unsigned long frequency = iter->second.value;

			// In grayscale, R = G = B, so just extract one of them
			unsigned char intensity = (unsigned char)(color >> 16); // Extracting red component
			frec_r[intensity] = frequency;
			frec_g[intensity] = frequency;
			frec_b[intensity] = frequency;
			if (frequency > max_frec) max_frec = frequency;
			//wxLogMessage("Intensity %u appears %lu times, eel test %li", intensity, frequency, frec_r[intensity]);
		}
		//wxLogMessage("Maximal value %li", max_frec);
	}
	Histogram* histT = new Histogram(max_frec, frec_r, frec_g, frec_b);
	return histT;

}

void cCanvas::OnMouseMove(wxMouseEvent& event) //----- FINISHED
{
	wxPoint pos = event.GetPosition();
	this->user_x = pos.x;
	this->user_y = pos.y;
}

void cCanvas::OnMouseClick(wxMouseEvent& event) // Esto aun no queda
{
	// primero deberia agregar el punto al vector de puntos, luego decremento de la variable
	if (this->points_left == -1) return;
	if (this->points_left == -3)
	{
		int* rgbU = this->getRGBPixel(event.GetPosition().x, event.GetPosition().y);
		Eigen::Vector3d vec;
		vec(0) = static_cast<double>(rgbU[0]); // declare the r value
		vec(1) = static_cast<double>(rgbU[1]); // declare the g value
		vec(2) = static_cast<double>(rgbU[2]); // declare the b value
		if (this->process.CmpNoCase("Euclidian") == 0)
		{
			//wxMessageBox("Se hara el proceso para distancia euclidiana");

			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			std::vector<double> distances = euclidean(matrixClasses, vec);
			int closest_class = getClosest(distances);
			wxString eucledian_message;
			eucledian_message.Printf(wxT("La clase mas cercana por distancia euclidiana es %d"), closest_class);
			wxMessageBox(eucledian_message);

		}
		if (this->process.CmpNoCase("mahalanobis") == 0)
		{
			//wxMessageBox("Se hara el proceso para distancia mahalanobis");
			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			std::vector<double> distances = manhalanobis(matrixClasses, vec);
			int closest_class = getClosest(distances);
			wxString manhalanobis_message;
			manhalanobis_message.Printf(wxT("La clase mas cercana por distancia mahalanobis es %d"), closest_class);
			wxMessageBox(manhalanobis_message);

		}
		if (this->process.CmpNoCase("MinProb") == 0)
		{
			//wxMessageBox("Se hara el proceso para distancia MaxProb");
			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			std::vector<double> probabilities = max_prob(matrixClasses, vec);
			int closest_class = getMaxProb(probabilities);
			wxString maxprob_message;
			maxprob_message.Printf(wxT("La clase mas cercana por criterio de maxima probabilidad, %d"), closest_class);
			wxMessageBox(maxprob_message);




		}


		if (this->process.CmpNoCase("KNN") == 0) {
			//wxMessageBox("Se hara el proceso para el criterio KNN");
			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			//std::vector<double> probabilities = max_prob(matrixClasses, vec);
			//int closest_class = getMaxProb(probabilities);

			int result = kNearestNeighbours(matrixClasses, vec, this->k);
			wxString knn_message;
			knn_message.Printf(wxT("La clase mas cercana por el criterio de KNN es %d"), result);
			wxMessageBox(knn_message);


		}
		delete[] rgbU;
		return;
	}
	if (this->points_left == -2)
	{
		this->points_left = -3;
		int* rgbU = this->getRGBPixel(event.GetPosition().x, event.GetPosition().y);
		Eigen::Vector3d vec;
		vec(0) = static_cast<double>(rgbU[0]); // declare the r value
		vec(1) = static_cast<double>(rgbU[1]); // declare the g value
		vec(2) = static_cast<double>(rgbU[2]); // declare the b value

		int minWidth = INT_MAX;
		int minHeight = INT_MAX;

		for (int i = 0; i < this->rectangles.size(); i += 2)
		{
			int width = this->rectangles[i + 1].x - this->rectangles[i].x;
			int height = this->rectangles[i + 1].y - this->rectangles[i].y;

			width = abs(width);
			height = abs(height);

			if (width < minWidth)
			{
				minWidth = width; // getting the min value
			}
			if (height < minHeight)
			{
				minHeight = height; // min value
			}
		}
		//wxString mess;
		//mess.Printf(wxT("Width más pequeño: %d, Height más pequeño: %d"), minWidth, minHeight);
		//wxMessageBox(mess);
		for (int i = 0; i < this->rectangles.size(); i += 2) // here is where we fill the matrix
		{
			// getting the upper left point (to iterate)
			int startX = std::min(this->rectangles[i].x, this->rectangles[i + 1].x);
			int startY = std::min(this->rectangles[i].y, this->rectangles[i + 1].y);
			Eigen::Matrix<double, Eigen::Dynamic, 3> matrix;
			for (int x = startX; x < startX + minWidth; ++x)
			{
				for (int y = startY; y < startY + minHeight; ++y)
				{
					matrix.conservativeResize(matrix.rows() + 1, Eigen::NoChange);

					auto rgb = pixelColors[std::make_pair(x, y)];

					matrix.row(matrix.rows() - 1) << std::get<0>(rgb),
						std::get<1>(rgb),
						std::get<2>(rgb);
				}
			}

			this->matrixClasses.push_back(matrix);
		}
		wxString messi;
		messi.Printf(wxT("%d, %d,% d  pixel a clasificar "), (int)vec(0), (int)vec(1), (int)vec(2));
		wxMessageBox(messi);
		/*for (auto a : this->matrixClasses) // the shape of all the matrix must be the same
		{
			wxString mess;
			mess.Printf(wxT("clases dimensiones %d, %d"), a.rows(), a.cols());
			wxMessageBox(mess);
		}
		*/
		// take the correct procedure
		if (this->process.CmpNoCase("Euclidian") == 0)
		{
			//wxMessageBox("Se hara el proceso para distancia euclidiana");

			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			std::vector<double> distances = euclidean(matrixClasses, vec);
			int closest_class = getClosest(distances);
			wxString eucledian_message;
			eucledian_message.Printf(wxT("La clase mas cercana por distancia euclidiana es %d"), closest_class);
			wxMessageBox(eucledian_message);

		}
		if (this->process.CmpNoCase("mahalanobis") == 0)
		{
			//wxMessageBox("Se hara el proceso para distancia mahalanobis");
			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			std::vector<double> distances = manhalanobis(matrixClasses, vec);
			int closest_class = getClosest(distances);
			wxString manhalanobis_message;
			manhalanobis_message.Printf(wxT("La clase mas cercana por distancia mahalanobis es %d"), closest_class);
			wxMessageBox(manhalanobis_message);

		}
		if (this->process.CmpNoCase("MinProb") == 0)
		{
			//wxMessageBox("Se hara el proceso para distancia MaxProb");
			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			std::vector<double> probabilities = max_prob(matrixClasses, vec);
			int closest_class = getMaxProb(probabilities);
			wxString maxprob_message;
			maxprob_message.Printf(wxT("La clase mas cercana por criterio de maxima probabilidad, %d"), closest_class);
			wxMessageBox(maxprob_message);




		}


		if (this->process.CmpNoCase("KNN") == 0) {
			//wxMessageBox("Se hara el proceso para el criterio KNN");
			// ------------CALL THE FUNCTION WITH THOSE VARIABLES AND SHOW THE INFO
			this->matrixClasses;// vector of each  matrix given a class
			vec; /// vec to compare

			//std::vector<double> probabilities = max_prob(matrixClasses, vec);
			//int closest_class = getMaxProb(probabilities);

			int result = kNearestNeighbours(matrixClasses, vec, k);
			wxString knn_message;
			knn_message.Printf(wxT("La clase mas cercana por el criterio de KNN es %d"), result);
			wxMessageBox(knn_message);


		}


		//this->rectangles.clear();
		//this->matrixClasses.clear();
		delete[] rgbU;
		//this->process = "";
		wxMessageBox(wxT("Click Derecho : Probar otro pixel \n CLick Izquierdo : Borrar clases "));
		Refresh();
		return;

	}
	this->rectangles.push_back(event.GetPosition());
	if ((this->points_left - 1) % 2 == 1) // When one rectangle its done
	{
		Refresh();
	}
	this->points_left = this->points_left - 1;
	if (this->points_left == 1)
	{
		wxMessageBox(wxT("Has Ingresado las clases, a continuacion ingresa el pixel el cual quieras clasificar "));
		this->points_left = -2;
		this->numClasses = 0;
		Refresh();
		return;
	}
	//wxMessageBox(wxT("Dibujando una clase"));
}
void cCanvas::OnMouseClickKill(wxMouseEvent& event)
{
	if (this->points_left == -1)
	{
		wxMessageBox(wxT("No hay mas clases que eliminar  "));
		return;
	}
	wxMessageBox(wxT("Se han eliminado las clases "));
	this->points_left = -1;
	this->rectangles.clear();
	this->matrixClasses.clear();
	//delete[] rgbU;
	this->process = "";
	Refresh();
}
int* cCanvas::getRGBPixel(int x, int y) {
	unsigned char redC = this->m_myImage[(this->getWidth() * (y - 1) + x) * 3];
	unsigned char greenC = this->m_myImage[(this->getWidth() * (y - 1) + x) * 3 + 1];
	unsigned char blueC = this->m_myImage[(this->getWidth() * (y - 1) + x) * 3 + 2];

	/*
	If we want to find the (2,2) position in this matrix, we obtain 9
		1 2 3
		4 5 6
		7 8 9
	Now, if thw matrix is lineal, we have something like
		1 2 3 4 5 6 7 8 9
	So
		WidthMatrix = 3
		Vecto[3*2+2-8]=9
	*/

	int* channels = new int[3];
	channels[0] = static_cast<int>(redC);
	channels[1] = static_cast<int>(greenC);
	channels[2] = static_cast<int>(blueC);

	return channels;
}
