#include "cMain.h"
#include "cEditorFrame.h"
#include "Histogram.h"
// our implementation of the GUI



wxBEGIN_EVENT_TABLE(cMain, wxMDIParentFrame)
// handle the event in the menu when is click (ID of the menuElement and the funciton to call in that event)
EVT_MENU(1001, cMain::InMenuOpenNew)
EVT_MENU(1002, cMain::InMenuSave)
EVT_MENU(1003, cMain::InMenuExit)
EVT_MENU(1004, cMain::InInfo)
EVT_MENU(2001, cMain::ToGray)
EVT_MENU(2002, cMain::ToRGB)
EVT_MENU(2003, cMain::ToHSV)
EVT_MENU(2004, cMain::ToHSL)
EVT_MENU(3001, cMain::getEuclidian) 
EVT_MENU(3002, cMain::getMahalanobis)
EVT_MENU(3003, cMain::getMinProb)
wxEND_EVENT_TABLE()

// The parent of all the components is wxMDIParentFrame
cMain::cMain() : wxMDIParentFrame(nullptr, wxID_ANY, "Image Wizard", wxPoint(30,30), wxSize(600,400 ))
{
	// menu bar  []
	wxMenuBar* menuBar = new wxMenuBar();
	this->SetMenuBar(menuBar);
	// adding sub-menus for the file menu [1001],[1002],[1003],[1004]
	wxMenu* menuFile = new wxMenu();
	menuFile->Append(1001, "OpenNew");
	menuFile->Append(1002, "Save Image");
	menuFile->Append(1004, "Info image");
	menuFile->Append(1003, "Exit");
	// ading the menu for the instance of the menuBar [ File[ New[1001], Open[1002], Save[1003], Exit[1004] ] ]
	menuBar->Append(menuFile, "File");
	// you got the idea...
	wxMenu* menuBf = new wxMenu();
	menuBf->Append(2001, "ToGray");
	menuBf->Append(2002, "ToRGB");
	menuBf->Append(2003, "ToHSV");
	menuBf->Append(2004, "ToHSL");
	menuBar->Append(menuBf, " Convert ");
	wxMenu* menuDis = new wxMenu();
	menuDis->Append(3001, "Euclidian");
	menuDis->Append(3002, "Mahalanobis");
	menuDis->Append(3003, "Min Prob");
	menuBar->Append(menuDis, " Distances ");

}
cMain:: ~cMain()
{
}
void cMain::ToGray(wxCommandEvent& event) 
{
	wxMDIChildFrame* child = this->GetActiveChild();
	if (child == nullptr)
	{
		wxMessageBox(wxT("You must open an image to conver to gray"));
		event.Skip();
		return;
	}
	cEditorFrame* mychild = wxDynamicCast(child, cEditorFrame);
	unsigned char* temp = mychild->getCanvas()->ToGray();
	if (!temp)
	{
		event.Skip();
		return;
	}
	cEditorFrame* process = new cEditorFrame(this, temp,mychild->GetName(), mychild->getCanvas()->getWidth(), mychild->getCanvas()->getHeight());
	process->Show();
	// nuevo constructor dado un unsigned char
	event.Skip();
}

void cMain::ToRGB(wxCommandEvent& event)
{
}

void cMain::ToHSV(wxCommandEvent& event)
{
}

void cMain::ToHSL(wxCommandEvent& event)
{
}
void cMain::getEuclidian(wxCommandEvent& event) // falta que manejes los eventos de guardar los rectangulos y dibujarlos jeje y maybe hacer una funcion q haga todo ese desma...
{
	wxMDIChildFrame* child = this->GetActiveChild();
	if (child == nullptr)
	{
		wxMessageBox(wxT("You must open an image to get a Distance"));
		event.Skip();
		return;
	}
	cEditorFrame* mychild = wxDynamicCast(child, cEditorFrame);
	mychild->getCanvas()->points_left = 0;
	wxMessageBox(wxT("Proceso para la distancia euclidiana"));
	// Crea un diálogo simple para la entrada numérica
	wxDialog dialog(this, wxID_ANY, wxT("Ingresa Numero de clases"), wxDefaultPosition, wxSize(250, 100));
	wxTextCtrl* numberEntry = new wxTextCtrl(&dialog, wxID_ANY, wxT(""), wxDefaultPosition, wxDefaultSize, 0, wxTextValidator(wxFILTER_NUMERIC));
	wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(numberEntry, 1, wxEXPAND | wxALL, 10);

	sizer->Add(dialog.CreateButtonSizer(wxOK | wxCANCEL), 0, wxEXPAND | wxALL, 5);

	dialog.SetSizer(sizer);
	dialog.SetAutoLayout(true);
	sizer->Fit(&dialog);

	if (dialog.ShowModal() == wxID_OK) {
		wxString numberStr = numberEntry->GetValue();
		long numberValue; 
		if (numberStr.ToLong(&numberValue))
		{
			int intValue = static_cast<int>(numberValue);
			mychild->getCanvas()->points_left = intValue*2;
		}
		else 
		{
			wxMessageBox(wxT("Por favor, ingrese un número válido."), wxT("Error"), wxOK | wxICON_ERROR);
		}
	}
	event.Skip();
}

void cMain::getMahalanobis(wxCommandEvent& event)
{
	wxMDIChildFrame* child = this->GetActiveChild();
	if (child == nullptr)
	{
		wxMessageBox(wxT("You must open an image to get a Distance"));
		event.Skip();
		return;
	}
	wxMessageBox(wxT("Proceso para la distancia mahalanobis"));
	event.Skip();
}

void cMain::getMinProb(wxCommandEvent& event)
{
	wxMDIChildFrame* child = this->GetActiveChild();
	if (child == nullptr)
	{
		wxMessageBox(wxT("You must open an image to get a Distance"));
		event.Skip();
		return;
	}
	wxMessageBox(wxT("Proceso para la distancia MinProb"));
	event.Skip();
}
void cMain::InMenuOpenNew(wxCommandEvent& event) // event to create a new window (wxMDIChildFrame)
{
	// Deberia crear una v
	wxBitmap bitmap;

	wxString filename = wxFileSelector(_T("Select file"), _T(""), _T(""), _T(""), _T("All files (*.*)|*.*"));
	if (!filename.empty())
	{
		cEditorFrame* window = new cEditorFrame(this, filename);
		window->Show();
		event.Skip();
	}
	else
	{
		wxMessageBox("Error loading the image", "Image Wizard", wxOK | wxICON_INFORMATION);
	}

		
	
}
void cMain::InMenuSave(wxCommandEvent& event)
{
	wxMDIChildFrame* child = this->GetActiveChild();
	if (child == nullptr)
	{
		wxMessageBox(wxT("You must open an image to save on your files"));
		event.Skip();
		return;
	}
	cEditorFrame* mychild = wxDynamicCast(child, cEditorFrame);
	if (!mychild->getCanvas()->img_load)
	{
		wxMessageBox(wxT("The window doesnt haave an image"));
		return;
	}
	wxString filename = wxFileSelector(_T("Save image as"), _T(""), _T(""), _T("*.bmp"), _T("BMP files (*.bmp)|*.bmp|GIF files (*gif)|*.gif|JPEG files (*jpg)|*.jpg|PNG files (*png)|*.png|TIFF files (*tif)|*.tif|XPM files (*xpm)|*.xpm|All files (*.*)|*.*"), wxFD_SAVE);
	if (!filename.empty())
	{
		mychild->getCanvas()->saveImage(filename);
	}
	

}
void cMain::InMenuExit(wxCommandEvent& event)
{
	Close();
	event.Skip();
}

void cMain::InInfo(wxCommandEvent& event)
{
	wxMDIChildFrame* child = this->GetActiveChild();
	if (child == nullptr)
	{
		wxMessageBox(wxT("You must open an image to see the info"));
		event.Skip();
		return;
	}
	cEditorFrame* mychild = wxDynamicCast(child, cEditorFrame);
	Histogram* h = mychild->getCanvas()->getHist();
	cEditorFrame* testing = new cEditorFrame(this, h);
	testing->Refresh();
	testing->Update();
	testing->Show();
	// nuevo constructor dado un unsigned char
	event.Skip();
}
