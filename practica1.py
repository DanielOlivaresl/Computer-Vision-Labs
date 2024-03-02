import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

markers = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]
colors = ["b", "g", "r", "c", "m", "y", "k", "w"] # Agrega mas colores por si son mas de 8 clases



def distance_euclidian(vec, data):
    mean = np.mean(data,axis=0)
    
    return np.sqrt(pow(mean[0]-vec[0],2) + pow(mean[1]-vec[1],2))


def distance_mahalanobis(vec, data):
    cov = np.cov(data.T)
    mean = np.mean(data, axis=0)
    x_minus_mu = vec - mean
    cov_inv = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, cov_inv)
    mahal_dist_sq = np.dot(left_term, x_minus_mu.T)
    return np.sqrt(mahal_dist_sq)

# def generate_class(dim, inter):
#     return np.random.randint(inter[0], inter[1], size=dim)

def plot_class(ax, data, color, marker):
    ax.scatter(data[:, 0], data[:, 1], color=color, marker=marker)

def calculate_and_plot(event):

    #Leemos los valores de las cajas de texto
    nc = text_box_nc.text
    ni = text_box_ni.text
    vecx = text_box_vecx.text
    vecy = text_box_vecy.text
    dis = text_box_dis.text

    # Genera valores aleatorios si los campos están vacíos
    nc = np.random.randint(1, 5) if not nc else int(nc)
    ni = np.random.randint(5, 13) if not ni else int(ni)
    vecx = np.random.uniform(-10, 10) if not vecx else float(vecx)
    vecy = np.random.uniform(-10, 10) if not vecy else float(vecy)
    dis = 2 if not dis else int(dis)

    #We calculate the missing classes

    while(nc>len(classes)):
        fill_class()

    #Vector del punto desconocido que queremos clasificar
    vec = np.array([vecx, vecy])

    





    main_ax.cla()  # Limpia el área de dibujo de los ejes principales.
    distances = []
    text = ""
    for i, cls in enumerate(classes):
        plot_class(main_ax, cls, color=colors[i % len(colors)], marker=markers[i % len(markers)])
        if dis == 2:
            dist = distance_mahalanobis(vec, cls)
            text = "Con distancia mahalanobis"
        else:
            dist= distance_euclidian(vec,cls)
            text = "Con distancia euclidiana"
        distances.append(dist)

    closest_class_index = np.argmin(distances)
    closest_class_color = colors[closest_class_index]

    main_ax.scatter(vec[0], vec[1], color="black", label="Punto de interés ")
    # Muestra la clase más cercana en la leyenda
    main_ax.text(0.5, -0.08, f"Clase más cercana: {closest_class_color} {text} (Clase {closest_class_index}), pos {vec[0]},{vec[1]}",
                 transform=main_ax.transAxes, ha="center", fontsize=10, color=colors[closest_class_index])
    main_ax.legend()




def fill_class():

    #This function will be called once the button to add a class is filled

    #This will read the content of the text boxes
    cent_x = text_box_cent_x.text
    cent_y = text_box_cent_y.text
    disp = disp_class.text

    ni = text_box_ni.text
    ni = np.random.randint(5, 13) if not ni else int(ni)

    nc= text_box_nc.text

    #We check if all classes have been calculated
    if nc and len(classes) >= int(nc):
        return
        



    #If the button was clicked and the cells were empty we generate the values randomly
    cent_x = np.random.normal(0,10,size=1) if not cent_x else float(cent_x)
    cent_y = np.random.normal(0,10,size=1) if not cent_y else float(cent_y)
    disp = np.random.uniform(0,10,1) if not disp else float(disp)
    
    x_values = np.random.normal(cent_x,disp,size=ni)
    y_values = np.random.normal(cent_y,disp,size=ni)

    classes.append(np.column_stack((x_values,y_values)))

def triggerFill(event):
    fill_class()
    text_box_cent_x.set_val('')
    text_box_cent_y.set_val('')
    disp_class.set_val('')
    



fig, main_ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

fig.set_size_inches(13,7)

#  cuadros de texto y sus etiquetas.
text_box_nc = TextBox(plt.axes([0.2, 0.2, 0.1, 0.05]), 'Num. Clases:')
text_box_ni = TextBox(plt.axes([0.2, 0.15, 0.1, 0.05]), 'Instancias por clase:')
text_box_vecx = TextBox(plt.axes([0.2, 0.1, 0.1, 0.05]), 'Posición X:')
text_box_vecy = TextBox(plt.axes([0.2, 0.05, 0.1, 0.05]), 'Posición Y:')
text_box_dis = TextBox(plt.axes([0.5, 0.2, 0.1, 0.05]), 'Distancia (1=Euc) (2=Mahalanobis):')

#Dispersion y clases 

text_box_cent_x= TextBox(plt.axes([0.5,0.15,0.1,0.05]), "Coordenada de centroide en X")
text_box_cent_y= TextBox(plt.axes([0.5,0.1,0.1,0.05]),"Coordenada de centroide en Y")
disp_class = TextBox(plt.axes([0.5,0.05,0.1,0.05]),"Dispersion de la clase") 


classes =[]

#Boton para llenar los centroides de las clases 
fill_button = Button(plt.axes([0.63,0.05,0.15,0.075]),'Llenar clase')
fill_button.on_clicked(triggerFill)



# botón para generar clases y visualizar informacion
#Left bottom width height
main_button = Button(plt.axes([0.8, 0.05, 0.15, 0.075]), 'Calcular y Graficar')
main_button.on_clicked(calculate_and_plot)



fig.canvas.manager.window.wm_geometry("+0+0")

plt.show()
