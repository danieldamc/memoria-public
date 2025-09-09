clc
close all 
clear

path = 'Procesados_MAT_FILES/';
files = dir(path);
mkdir('Procesados_MAT_FILES_mov/')


for n=1:size(files,1)
    if length(files(n).name)>3
        

        load([path,files(n).name])
        r = randperm(size(img_list_new,3)); % permutacion aleatoria de 1 al numero de cortes.
        img_list_new_out = img_list_new;

        % voy a mover el 60% de los cortes
        cant_datos = round(length(r)*0.6);
        slices_seleccionados = r(1:cant_datos);

        % para mover los slices debo seleccionar la direccion en la cual lo
        % voy a mover, el 50% de los slices seleccionados los movere en las
        % filas, y el resto en las columnas.
        cant_datos = round(length(slices_seleccionados)*0.5);
        slices_seleccionados_filas = slices_seleccionados(1:cant_datos);
        slices_seleccionados_columnas = slices_seleccionados(cant_datos+1:end);

        % Con respecto al desplazamiento este sera como maximo de 15% del 
        % y un minimo de 5% del tamaño de la imagen en la direccion analizada.  
        numero_filas_15 = round(size(img_list_new,1)*0.05);
        numero_columns_15 = round(size(img_list_new,2)*0.05);
        numero_filas_5 = round(size(img_list_new,1)*0.01);
        numero_columns_5 = round(size(img_list_new,2)*0.01);
        
        desplazamiento_filas = round((randperm(numero_filas_15)./numero_filas_15)*(numero_filas_15-numero_filas_5) + numero_filas_5);
        desplazamiento_columnas = round((randperm(numero_columns_15)./numero_columns_15)*(numero_columns_15-numero_columns_5) + numero_columns_5);
        
        desplazamiento_seleccionados_filas = desplazamiento_filas(1:length(slices_seleccionados_filas));
        desplazamiento_seleccionados_columnas = desplazamiento_columnas(1:length(slices_seleccionados_columnas));

        for k = 1:length(slices_seleccionados_filas)
            se = translate(strel(1), [desplazamiento_seleccionados_filas(k) 0]);
            img_list_new_out(:,:,slices_seleccionados_filas(k)) = imdilate(img_list_new(:,:,slices_seleccionados_filas(k)),se);
        end

        for k = 1:length(slices_seleccionados_columnas)
            se = translate(strel(1), [0 desplazamiento_seleccionados_columnas(k)]);
            img_list_new_out(:,:,slices_seleccionados_columnas(k)) = imdilate(img_list_new(:,:,slices_seleccionados_columnas(k)),se);
        end

        img_list_new = img_list_new_out;
        save(['Procesados_MAT_FILES_mov/',files(n).name(1:end-4),'_mov.mat'],'img_list_new')
    end
end