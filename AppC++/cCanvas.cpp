#include "cCanvas.h"
#include <string>
#include <fstream>
#include "Histogram.h"
#include <Eigen/Dense>

wxBEGIN_EVENT_TABLE(cCanvas, wxHVScrolledWindow)
EVT_PAINT(cCanvas::OnPaint)
wxEND_EVENT_TABLE()

// agregar logica para un nuevo constructor dado un histograma y que en la funcion de dibujar compruebe si sse va a dibujar un histograma 

// mandarle la imagen de el histograma y solamente dibujar mas 
cCanvas::cCanvas(wxWindow* parent, wxString filename) : wxHVScrolledWindow(parent, wxID_ANY)  // ----- FINISHED
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	Bind(wxEVT_RIGHT_UP, &cCanvas::OnMouseClick, this);
	SetRowColumnCount(40, 40); // see the changes if change that values
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	this->fileName = filename;
	this->img_load = true;
}

cCanvas::cCanvas(wxWindow* parent, unsigned char* data, int w, int h) : wxHVScrolledWindow(parent, wxID_ANY) // constuctor given a new image - FINISHED
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	Bind(wxEVT_RIGHT_UP, &cCanvas::OnMouseClick, this);
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

				// Establecer el pincel como transparente para no rellenar el rectángulo
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
			unsigned char green = this->m_myImage[offset + 1]- 0;
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
	else if(this->getformat() == (wxString)"GRAY")
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
	if (this->points_left == -2)
	{
		this->points_left = -1;
		wxString message;
		message.Printf(wxT("vector en (%d, %d)"), event.GetPosition().x, event.GetPosition().y);
		wxMessageBox(message);
		int minWidth = INT_MAX; // Inicializa minWidth a un valor muy grande
		int minHeight = INT_MAX; // Inicializa minHeight a un valor muy grande

		for (int i = 0; i < this->rectangles.size(); i += 2)
		{
			int width = this->rectangles[i + 1].x - this->rectangles[i].x;
			int height = this->rectangles[i + 1].y - this->rectangles[i].y;

			// Asegúrate de que width y height sean positivos
			width = abs(width);
			height = abs(height);

			if (width < minWidth) {
				minWidth = width; // Actualiza minWidth si se encuentra un width más pequeño
			}
			if (height < minHeight) {
				minHeight = height; // Actualiza minHeight si se encuentra un height más pequeño
			}
		}

		// Ahora minWidth y minHeight contienen los valores más pequeños encontrados
		wxString mess;
		mess.Printf(wxT("Width más pequeño: %d, Height más pequeño: %d"), minWidth, minHeight);
		wxMessageBox(mess);
		for (int i = 0; i < this->rectangles.size(); i += 2) {
			// Determina la esquina superior izquierda
			int startX = std::min(this->rectangles[i].x, this->rectangles[i + 1].x);
			int startY = std::min(this->rectangles[i].y, this->rectangles[i + 1].y);

			// Itera desde la esquina superior izquierda hasta startX + minWidth y startY + minHeight
			for (int x = startX; x < startX + minWidth; ++x) {
				for (int y = startY; y < startY + minHeight; ++y) 
				{
				 // llamar a la funcion y agregar a la funcion 
				}
			}
		}

		// cadena de ifs para comprobar el proceso
		if (this->process.CmpNoCase("Euclidian") == 0)
		{
			wxMessageBox("Se hara el proceso para distancia euclidiana");
		}
		this->rectangles.clear();
		Refresh();
		return;

	}
	this->rectangles.push_back(event.GetPosition());
	if ((this->points_left-1) % 2 == 1)
	{
		//wxMessageBox(wxT("Se dibujo el rectangulo"));
		Refresh();
	}
	this->points_left = this->points_left - 1;
	if (this->points_left == 1)
	{
		wxMessageBox(wxT("se han acabdo las clases a graficar"));
		this->points_left = -2;
		this->numClasses = 0;
		Refresh();
		return;
	}
	//wxMessageBox(wxT("Dibujando una clase"));
}


	
