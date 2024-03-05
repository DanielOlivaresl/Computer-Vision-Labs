#pragma once
#include "wx/wx.h"
#include "wx/vscroll.h"
#include "wx/dcclient.h"
#include "wx/dcmemory.h"
#include "wx/dcbuffer.h"
#include "Histogram.h"
#include "vector"
// add a general function for convert that can iterate in the image an given a parameter (the format), apply that format
// conver(RGB,HSL,HSV,GRAY)



// canvas in wich wee can edite the sprite
class cCanvas : public wxHVScrolledWindow
{
public: 
	cCanvas(wxWindow* parent, wxString filename);
	cCanvas(wxWindow* parent, unsigned char* data, int w, int h);
	cCanvas(wxWindow* parent, Histogram* h);
	int getWidth();
	int getHeight();
	~cCanvas();
	wxString fileName; // file name of the image // i can handle better this
	wxString fileFormat; // format of the file 
	bool img_load; // if theres is an image loaded
	Histogram* hist = nullptr;
	
private: 
	int m_nPixelSize = 8; // for the slider
	
	//----- Image variables---------------------------------------------------------------------
	int m_imageWidth;                                                                         //
	int m_imageHeight;                                                                        //
	wxBitmap m_imageBitmap;	// used to display the image                                      //
	wxImage* m_imageRGB = nullptr;		// used to load the image                              //
	unsigned char* m_myImage = nullptr;	// used to process the image (raw data) pixels in 1d  //
	//-----------------------------------------------------------------------------------------
public:
	//---- Process data
	int user_x = 0;
	int user_y = 0; // this variables always have the user position
	int points_left = -1; // -1 if there is no process, 0 if the user draw all the rectangles (trigger to calculate and show)
	// Agregar el vector de puntos del usuario, los rectangulos de cada clase
	
	//--------Image functions--------------------------------------------------------------------------------------------

	void LoadImage(); // for load the image and set m_imageHeight , m_imageWidth, m_imageBitmap, m_imageRGB, m_myImage //
	void OnPaint(wxPaintEvent& event); // for init the draw context                                                    //
	void OnDraw(wxDC& dc); // actually drawing  
	unsigned char* ToGray();// return the gray format of the image
	wxString getformat(); // returns the format of the image       
	void saveImage(wxString filename);//
	Histogram* getHist();

	
	//-------------------------------------------------------------------------------------------------------------------
	
	void setPixelSize(int p); // for the slider
private:
	virtual wxCoord OnGetRowHeight(size_t row) const; // for the slider
	virtual wxCoord OnGetColumnWidth(size_t col)const; // for the slider
	void OnMouseMove(wxMouseEvent& event); // function to get the user position in the canvas
	//void OnMouseClick(wxMouseEvent& event); // function to draw rectangles
	wxDECLARE_EVENT_TABLE();

};

