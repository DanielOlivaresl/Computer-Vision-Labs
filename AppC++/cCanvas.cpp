#include "cCanvas.h"
#include <string>
#include <fstream>
#include "Histogram.h"
wxBEGIN_EVENT_TABLE(cCanvas, wxHVScrolledWindow)
EVT_PAINT(cCanvas::OnPaint)
wxEND_EVENT_TABLE()

// agregar logica para un nuevo constructor dado un histograma y que en la funcion de dibujar compruebe si sse va a dibujar un histograma 

// mandarle la imagen de el histograma y solamente dibujar mas 
cCanvas::cCanvas(wxWindow* parent, wxString filename) : wxHVScrolledWindow(parent, wxID_ANY)
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	SetRowColumnCount(40, 40); // see the changes if change that values
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	this->fileName = filename;
	this->img_load = true;
}

cCanvas::cCanvas(wxWindow* parent, unsigned char* data, int w, int h) : wxHVScrolledWindow(parent, wxID_ANY) // constuctor given a new image
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	SetRowColumnCount(40, 40); // see the changes if change that values
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	this->m_myImage = data;
	this->m_imageWidth = w;
	this->m_imageHeight = h;
	this->m_imageRGB = new wxImage(this->m_imageWidth, this->m_imageHeight, this->m_myImage, true);
	this->img_load = true;
}

cCanvas::cCanvas(wxWindow* parent, Histogram* h) // constructor to create a histogram given a image 
{
	Bind(wxEVT_MOTION, &cCanvas::OnMouseMove, this);
	SetBackgroundStyle(wxBG_STYLE_PAINT);
	this->hist = h;
	this->m_imageRGB = nullptr;
	this->m_myImage = nullptr;
	this->img_load = false;
	this->m_imageHeight = 0;
	this->m_imageWidth = 0;

}

int cCanvas::getWidth()
{
	return this->m_imageWidth;
}

int cCanvas::getHeight()
{
	return this->m_imageHeight;
}

cCanvas::~cCanvas()
{
	delete this->hist;
}

void cCanvas::setPixelSize(int p)
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
		//dc.DrawRectangle(10, 10, 150, 100);
	}
	else
	{
		dc.DrawRectangle(10, 10, 100, 50);
		wxMessageBox(wxT("Entramos al hist"));
	}
		
}
unsigned char* cCanvas::ToGray() // testing...
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
wxString cCanvas::getformat()
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
void cCanvas::saveImage(wxString filename)
{
	bool b;

	wxImage* tempImage = new wxImage(this->m_imageWidth, this->m_imageHeight, this->m_myImage, true); // lend my image buffer...
	b = tempImage->SaveFile(filename);
	delete(tempImage);		// buffer not needed any more

	if (!b)
		wxMessageBox(wxT("A problem occured during saving"));
}
void cCanvas::LoadImage()
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
wxCoord cCanvas::OnGetRowHeight(size_t row) const
{
	return wxCoord(m_nPixelSize);
}

wxCoord cCanvas::OnGetColumnWidth(size_t col) const
{
	return wxCoord(m_nPixelSize);
}
Histogram* cCanvas::getHist()
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

void cCanvas::OnMouseMove(wxMouseEvent& event)
{
	wxPoint pos = event.GetPosition();
	this->user_x = pos.x;
	this->user_y = pos.y;
}

	
