// Entry point of the app

#include "cApp.h"

#define APP_NAME "Image Wizard" // defining the app name

// Implemented... there is no more to add in this part

wxIMPLEMENT_APP(cApp); 

cApp::cApp()
{
}

cApp::~cApp()
{
}

bool cApp::OnInit() // Init of the app,,,, like the main in c
{
	wxInitAllImageHandlers(); // To suport all the image files
	m_frame1 = new cMain(); // Creating a new instance of a wx MDI parent to suport multiple window handle (the main window)
	m_frame1->Show(); // window show
	return true;
}
